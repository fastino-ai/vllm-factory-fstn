"""GLiNER 2.5 parity: hosted vLLM output against local GLiNER2 batch APIs.

Two-phase design:
    Phase 1 (--prepare): local BoundaryExtractor / Classifier / JointIEEngine
        references plus a vLLM model dir. Install GLiNER2 at PR #165 head
        (af36b41, version 2.1.0) before running.
    Phase 2 (--test): vLLM inference compared to those references.

Target: vllm==0.20.0, checkpoint fastino/gliner2.5-multi-v1 (334 head tensors).

Usage:
    python plugins/deberta_gliner25/parity_test.py --prepare
    python plugins/deberta_gliner25/parity_test.py --test
"""

from __future__ import annotations

import argparse
import json
import os

MODEL = "fastino/gliner2.5-multi-v1"
LOCAL_MODEL_DIR = "/tmp/gliner25-multi-vllm"
REF_FILE = "/tmp/gliner25-multi-reference.json"

TEXT = (
    "John Smith works at NVIDIA Corporation in Santa Clara, California. "
    "His email is john.smith@nvidia.com and phone number is 555-123-4567. "
    "He is the VP of AI Research and reports to Jensen Huang."
)

SCHEMA = {
    "entities": {
        "person": "",
        "organization": "",
        "location": "",
        "email": "",
        "phone_number": "",
    },
    "classifications": [
        {
            "task": "topic",
            "labels": ["technology", "finance", "sports", "healthcare"],
        }
    ],
    "relations": {"works_at": "", "reports_to": ""},
    "structures": {
        "employee": {
            "fields": [
                {"name": "name", "dtype": "str"},
                {"name": "title", "dtype": "str"},
                {"name": "company", "dtype": "str"},
            ]
        }
    },
}

THRESHOLD = 0.5
EXPECTED_HEAD_TENSORS = 334

# Hosted output must match a local GLiNER2 call on these payloads.
CASES: list[dict] = [
    {
        "name": "classic",
        "runtime": "classic",
        "text": TEXT,
        "schema": SCHEMA,
        "threshold": THRESHOLD,
        "include_confidence": True,
        "include_spans": True,
    },
    {
        "name": "classic_thresholds",
        "runtime": "classic",
        "text": TEXT,
        "schema": {
            "entities": {
                "person": {"threshold": 0.9},
                "organization": {"threshold": 0.1},
            },
            "classifications": [
                {
                    "task": "topic",
                    "labels": ["technology", "finance", "sports", "healthcare"],
                    "cls_threshold": 0.8,
                }
            ],
        },
        "threshold": THRESHOLD,
        "include_confidence": True,
        "include_spans": True,
    },
    {
        "name": "records",
        "runtime": "classic",
        "text": "Alice met Bob in Paris. Carol met Dave in London.",
        "schema": {
            "structures": {
                "meeting": {
                    "mode": "natural",
                    "anchor": "person",
                    "fields": [
                        {"name": "person", "dtype": "str"},
                        {"name": "city", "dtype": "str"},
                    ],
                }
            }
        },
        "threshold": THRESHOLD,
        "include_confidence": True,
        "include_spans": True,
    },
    {
        "name": "span_attributes",
        "runtime": "classic",
        "text": "Alice was delighted, but Bob sounded frustrated.",
        "schema": {
            "entities": ["person"],
            "entity_attributes": {
                "sentiment": {
                    "labels": ["positive", "negative", "neutral"],
                    "applies_to": ["person"],
                    "qualify_labels": True,
                }
            },
        },
        "threshold": THRESHOLD,
        "include_confidence": True,
        "include_spans": True,
    },
    {
        "name": "constrained_classification",
        "runtime": "constrained_classification",
        "text": "Delete the temporary file",
        "schema": {
            "tasks": {
                "intent": {
                    "labels": ["read", "write", "delete"],
                    "min_labels": 1,
                    "max_labels": 1,
                },
                "effects": {
                    "labels": ["read_only", "create", "modify", "delete"],
                    "min_labels": 1,
                },
            },
            "constraints": [
                {
                    "type": "Implies",
                    "cond": {"type": "LabelRef", "task": "intent", "label": "delete"},
                    "then": {"type": "LabelRef", "task": "effects", "label": "delete"},
                }
            ],
        },
        "include_confidence": True,
        "include_spans": False,
    },
    {
        "name": "joint_ie",
        "runtime": "joint_ie",
        "text": "Alice works for Acme. Bob joined Acme last year.",
        "schema": {
            "entities": ["person", "organization"],
            "relations": {
                "works_for": {
                    "head": "person",
                    "tail": "organization",
                    "unique_head": True,
                }
            },
        },
        "include_confidence": True,
        "include_spans": True,
    },
]
# bf16 on the vLLM path against fp32 eager in AutoExtractor.
CONFIDENCE_TOLERANCE = 0.05


def _entity_spans(payload: dict) -> dict[str, list[tuple]]:
    """Map each entity type to its extracted ``(text, start, end)`` spans."""
    spans: dict[str, list[tuple]] = {}
    for label, records in (payload.get("entities") or {}).items():
        found = []
        for record in records:
            if isinstance(record, dict):
                found.append(
                    (record.get("text"), record.get("start"), record.get("end"))
                )
            else:
                found.append((record, None, None))
        spans[label] = found
    return spans


def _entity_confidences(payload: dict) -> dict[tuple, float]:
    """Map each ``(type, text)`` to its confidence, where one was returned."""
    scores: dict[tuple, float] = {}
    for label, records in (payload.get("entities") or {}).items():
        for record in records:
            if isinstance(record, dict) and "confidence" in record:
                scores[label, record.get("text")] = float(record["confidence"])
    return scores


def _classifications(payload: dict) -> dict[str, str]:
    """Map each classification task to its winning label."""
    picked: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and "label" in value:
            picked[key] = str(value["label"])
        elif isinstance(value, str) and key != "text":
            picked[key] = value
    return picked


def compare_outputs(
    reference: dict, candidate: dict, *, tolerance: float = CONFIDENCE_TOLERANCE
) -> list[str]:
    """Diff a vLLM result against the AutoExtractor reference.

    Args:
        reference: Formatted AutoExtractor output.
        candidate: Formatted vLLM output for the same text and schema.
        tolerance: Largest confidence difference treated as agreement.

    Returns:
        One human-readable line per disagreement; empty means parity.
    """
    problems: list[str] = []
    ref_spans, got_spans = _entity_spans(reference), _entity_spans(candidate)
    for label in sorted(set(ref_spans) | set(got_spans)):
        expected, actual = ref_spans.get(label, []), got_spans.get(label, [])
        if expected != actual:
            problems.append(f"entities[{label}]: expected {expected}, got {actual}")

    ref_scores, got_scores = (
        _entity_confidences(reference),
        _entity_confidences(candidate),
    )
    for key in sorted(set(ref_scores) & set(got_scores)):
        delta = abs(ref_scores[key] - got_scores[key])
        if delta > tolerance:
            problems.append(
                f"confidence{list(key)}: {ref_scores[key]:.4f} vs "
                f"{got_scores[key]:.4f} (delta {delta:.4f} > {tolerance})"
            )

    ref_cls, got_cls = _classifications(reference), _classifications(candidate)
    for task in sorted(set(ref_cls) | set(got_cls)):
        if ref_cls.get(task) != got_cls.get(task):
            problems.append(
                f"classification[{task}]: expected {ref_cls.get(task)!r}, got {got_cls.get(task)!r}"
            )

    skip = {"text"}
    ref_keys = set(reference) - skip
    got_keys = set(candidate) - skip
    if ref_keys != got_keys:
        problems.append(f"keys: expected {sorted(ref_keys)}, got {sorted(got_keys)}")
    return problems


def _flags(case: dict) -> dict:
    """Return the request flags a case carries into GLiNER2."""
    return {
        "include_confidence": bool(case.get("include_confidence", False)),
        "include_spans": bool(case.get("include_spans", False)),
        "threshold": float(case.get("threshold", THRESHOLD)),
        "max_len": case.get("max_len"),
    }


def local_output(extractor: object, classifier: object, joint: object, case: dict) -> dict:
    """Run one case through the local GLiNER2 API for its runtime.

    Args:
        extractor: Encoder-backed ``BoundaryExtractor``.
        classifier: ``Classifier`` wrapping that extractor.
        joint: ``JointIEEngine`` wrapping that extractor.
        case: One entry of ``CASES``.

    Returns:
        The JSON-ready result dict.
    """
    from gliner2.classification import ClassificationConfig, ClassificationSchema
    from gliner2.inference.schema import Schema
    from gliner2.joint_ie import JointIEConfig
    from gliner2.joint_ie.schema import JointSchema

    flags = _flags(case)
    text = case["text"]
    schema = case["schema"]
    runtime = case["runtime"]
    if runtime == "classic":
        return extractor.batch_extract(
            [text],
            [Schema.from_dict(schema)],
            threshold=flags["threshold"],
            include_confidence=flags["include_confidence"],
            include_spans=flags["include_spans"],
            max_len=flags["max_len"],
        )[0]
    if runtime == "constrained_classification":
        config = ClassificationConfig(
            include_confidence=flags["include_confidence"],
            max_len=flags["max_len"],
        )
        result = classifier.batch_classify(
            [text],
            [ClassificationSchema.from_dict(schema)],
            config=config,
        )[0]
        return result.to_dict()
    config = JointIEConfig(
        include_confidence=flags["include_confidence"],
        include_spans=flags["include_spans"],
        max_len=flags["max_len"],
    )
    result = joint.batch_extract(
        [text], [JointSchema.from_dict(schema)], config=config
    )[0]
    return result.to_dict()


def compare_json(
    reference: object,
    candidate: object,
    *,
    tolerance: float = CONFIDENCE_TOLERANCE,
    path: str = "$",
) -> list[str]:
    """Diff two JSON-like values, allowing small float drift.

    Args:
        reference: Local GLiNER2 output.
        candidate: Hosted vLLM output.
        tolerance: Largest absolute float difference treated as agreement.
        path: Location of this value, for mismatch lines.

    Returns:
        One line per disagreement.
    """
    if isinstance(reference, dict) and isinstance(candidate, dict):
        problems: list[str] = []
        for key in sorted(set(reference) | set(candidate)):
            if key not in reference or key not in candidate:
                problems.append(f"{path}.{key}: missing on one side")
                continue
            problems.extend(
                compare_json(reference[key], candidate[key], tolerance=tolerance, path=f"{path}.{key}")
            )
        return problems
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            return [f"{path}: length {len(reference)} vs {len(candidate)}"]
        problems = []
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            problems.extend(compare_json(left, right, tolerance=tolerance, path=f"{path}[{index}]"))
        return problems
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        if isinstance(reference, bool) or isinstance(candidate, bool):
            if reference is not candidate:
                return [f"{path}: {reference!r} vs {candidate!r}"]
            return []
        if abs(float(reference) - float(candidate)) > tolerance:
            return [f"{path}: {reference} vs {candidate}"]
        return []
    if reference != candidate:
        return [f"{path}: {reference!r} vs {candidate!r}"]
    return []


def phase_prepare(
    model_name: str = MODEL,
    local_model_dir: str = LOCAL_MODEL_DIR,
    ref_file: str = REF_FILE,
) -> None:
    from gliner2 import AutoExtractor
    from gliner2.classification import Classifier
    from gliner2.joint_ie import JointIEEngine

    from forge.model_prep import prepare_gliner25_model

    print("=" * 60)
    print(f"PHASE 1: local GLiNER2 references ({model_name})")
    print("=" * 60)

    extractor = AutoExtractor.from_pretrained(model_name)
    extractor.eval()
    classifier = Classifier(extractor)
    joint = JointIEEngine(extractor)
    references = []
    for case in CASES:
        output = local_output(extractor, classifier, joint, case)
        print(f"--- {case['name']} ---")
        print(json.dumps(output, indent=2, default=str)[:2000])
        references.append({"name": case["name"], "case": case, "output": output})

    state = extractor.state_dict()
    head_keys = [
        k
        for k in state
        if k.startswith(
            ("boundary_head.", "record_decoder.", "relation_scorer.", "classifier.")
        )
    ]
    print(f"Head tensors: {len(head_keys)}")
    if len(head_keys) != EXPECTED_HEAD_TENSORS:
        raise SystemExit(
            f"Expected {EXPECTED_HEAD_TENSORS} head tensors, got {len(head_keys)}"
        )

    os.makedirs(os.path.dirname(ref_file) or ".", exist_ok=True)
    with open(ref_file, "w") as f:
        json.dump({"model": model_name, "cases": references}, f, default=str)

    prepared = prepare_gliner25_model(
        model_name, output_dir=local_model_dir, force=True
    )
    print(f"Prepared model dir: {prepared}")
    print("Phase 1 complete")


def phase_test(
    model_name: str = MODEL,
    local_model_dir: str = LOCAL_MODEL_DIR,
    ref_file: str = REF_FILE,
) -> bool:
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from vllm.pooling_params import PoolingParams

    from plugins.deberta_gliner25.processor import (
        decode_boundary_output,
        preprocess_boundary,
        prompt_schema,
    )

    print("=" * 60)
    print(f"PHASE 2: vLLM inference + parity ({model_name})")
    print("=" * 60)

    with open(ref_file) as f:
        ref = json.load(f)
    saved = {item["name"]: item for item in ref["cases"]}

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    llm = LLM(
        model=local_model_dir,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.78,
    )
    problems: list[str] = []
    for case in CASES:
        flags = _flags(case)
        prompt = prompt_schema(case["runtime"], case["schema"])
        prep = preprocess_boundary(
            tokenizer,
            case["text"],
            case["schema"],
            prompt=prompt,
            runtime=case["runtime"],
            threshold=flags["threshold"],
            include_confidence=flags["include_confidence"],
            include_spans=flags["include_spans"],
        )
        pooling_params = PoolingParams(task="plugin", extra_kwargs=prep["extra_kwargs"])
        tokens = TokensPrompt(prompt_token_ids=prep["input_ids"])
        outputs = llm.encode([tokens], pooling_params=pooling_params, pooling_task="plugin")
        hosted = decode_boundary_output(outputs[0].outputs.data, case["schema"])
        reference = saved[case["name"]]["output"]
        case_problems = compare_json(reference, hosted)
        for problem in case_problems:
            problems.append(f"{case['name']}: {problem}")
        print(f"--- {case['name']} ---")
        print(json.dumps(hosted, indent=2, default=str)[:2000])
    for problem in problems:
        print(f"  MISMATCH {problem}")
    print("PASS" if not problems else f"FAIL ({len(problems)} mismatch(es))")
    return not problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if not args.prepare and not args.test:
        phase_prepare()
        ok = phase_test()
        raise SystemExit(0 if ok else 1)
    if args.prepare:
        phase_prepare()
    if args.test:
        ok = phase_test()
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
