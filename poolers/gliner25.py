"""GLiNER 2.5 boundary pooler.

vLLM runs the encoder. This module groups sequences and hands the hidden
states to GLiNER2's own batch APIs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
import torch.nn as nn

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states

logger = logging.getLogger(__name__)

RUNTIMES = ("classic", "constrained_classification", "joint_ie")


def _require(module: str, purpose: str):
    from vllm_factory.optional_deps import require

    return require(module, "gliner2", purpose=purpose)


def _group_key(extra: dict[str, Any]) -> tuple[Any, ...]:
    """Return the settings that must match for one GLiNER2 batch call.

    Args:
        extra: Compact extras for one sequence.

    Returns:
        ``(runtime, threshold, include_confidence, include_spans, max_len)``.
    """
    max_len = extra.get("max_len")
    return (
        str(extra.get("runtime") or "classic"),
        float(extra.get("threshold", 0.5)),
        bool(extra.get("include_confidence", False)),
        bool(extra.get("include_spans", False)),
        None if max_len is None else int(max_len),
    )


def _sequence_groups(
    ctx: PoolerContext, extras: list[dict[str, Any]]
) -> list[list[int]]:
    """Group sequence indexes that can share one GLiNER2 call.

    Args:
        ctx: Scheduled batch context, including adapter slots.
        extras: Per-sequence extras aligned with the scheduled batch.

    Returns:
        Groups of indexes. A mixed-adapter batch is one sequence per group.
        Empty extras are omitted.
    """
    pending = [index for index, extra in enumerate(extras) if extra]
    if not ctx.shares_one_adapter():
        return [[index] for index in pending]
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index in pending:
        grouped.setdefault(_group_key(extras[index]), []).append(index)
    return list(grouped.values())


class GLiNER25BoundaryPooler(nn.Module):
    """Encoder-less GLiNER2 boundary model plus the two non-classic facades."""

    def __init__(self, *, config: Any, encoder_config: Any, tokenizer: Any) -> None:
        """Build heads once from the checkpoint config.

        Args:
            config: GLiNER2 ``ExtractorConfig`` with ``architecture="boundary"``.
            encoder_config: Config whose ``hidden_size`` matches the vLLM encoder.
            tokenizer: Tokenizer the extractor collates with.
        """
        super().__init__()
        engine = _require("gliner2.inference.engine", "GLiNER25 boundary extractor")
        classification = _require("gliner2.classification", "GLiNER25 classifier")
        joint = _require("gliner2.joint_ie", "GLiNER25 joint IE")
        self.extractor = engine.BoundaryExtractor(
            config,
            encoder_config=encoder_config,
            tokenizer=tokenizer,
            load_encoder=False,
        )
        self.extractor.eval()
        self.task_classifier = classification.Classifier(self.extractor)
        self.joint_engine = joint.JointIEEngine(self.extractor)
        self._runtime_device: tuple[torch.device, torch.dtype] | None = None

    def get_tasks(self) -> set[str]:
        """Return the pooling tasks this pooler serves."""
        return {"embed", "classify", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        """Decode every sequence in one scheduled batch.

        Args:
            hidden_states: Concatenated encoder output for the whole batch.
            ctx: Scheduled batch context — sequence lengths and per-prompt
                extras, in the same order.

        Returns:
            One packed JSON payload per scheduled sequence.
        """
        extras = list(ctx.extra_kwargs)
        device = hidden_states.device
        try:
            sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        except (IndexError, TypeError, RuntimeError):
            logger.exception("[GLiNER25] cannot split hidden states; returning empty")
            return _empty_payloads(len(extras) or 1, device)

        if len(extras) < len(sequences):
            extras.extend({} for _ in range(len(sequences) - len(extras)))
        else:
            extras = extras[: len(sequences)]

        outputs: list[torch.Tensor | None] = [None] * len(sequences)
        for indices in _sequence_groups(ctx, extras):
            with ctx.lora_scope(indices[0]):
                packed = self._decode_group(
                    [sequences[index] for index in indices],
                    [extras[index] for index in indices],
                )
            if len(packed) != len(indices):
                raise RuntimeError(
                    f"decode returned {len(packed)} payloads for {len(indices)} sequences"
                )
            for index, payload in zip(indices, packed, strict=True):
                outputs[index] = payload
        for index, extra in enumerate(extras):
            if not extra:
                outputs[index] = _pack_json({}, sequences[index].device)
        if any(payload is None for payload in outputs):
            raise RuntimeError("pooler left a sequence without a payload")
        return outputs

    def _decode_group(
        self,
        sequences: list[torch.Tensor],
        extras: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        """Run one GLiNER2 batch call and pack each result.

        Args:
            sequences: Per-sequence hidden states, one ``(tokens, hidden)`` tensor.
            extras: Compact extras sharing one group key, same order.

        Returns:
            One packed JSON payload per sequence.

        Raises:
            ValueError: An extra is not compact, or its runtime is unknown.
        """
        from plugins.deberta_gliner25.processor import is_compact_extra

        for extra in extras:
            if not is_compact_extra(extra):
                raise ValueError(f"expected compact extra_kwargs, got keys {sorted(extra)[:8]}")
        sample = sequences[0]
        self._bind_runtime(sample)
        with torch.inference_mode():
            results = self._run_runtime(sequences, extras)
        return [_pack_json(result, sample.device) for result in results]

    def _bind_runtime(self, sample: torch.Tensor) -> None:
        """Point both facades at ``sample``'s device the first time it changes.

        Args:
            sample: One hidden-state row from the scheduled batch.
        """
        placed = (sample.device, sample.dtype)
        if self._runtime_device == placed:
            return
        self.task_classifier.to(device=sample.device, dtype=sample.dtype)
        self.joint_engine.to(device=sample.device, dtype=sample.dtype)
        self._runtime_device = placed

    def _run_runtime(
        self,
        sequences: list[torch.Tensor],
        extras: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Dispatch a group to the runtime's GLiNER2 batch API.

        Args:
            sequences: Per-sequence hidden states.
            extras: Compact extras for those sequences.

        Returns:
            One result dict per sequence.

        Raises:
            ValueError: ``runtime`` is not a hosted GLiNER2 runtime.
        """
        runtime, threshold, include_confidence, include_spans, max_len = _group_key(extras[0])
        texts = [str(extra["text"]) for extra in extras]
        schemas = [extra["schema"] for extra in extras]
        if runtime == "classic":
            return self._run_classic(
                texts, schemas, sequences, threshold, include_confidence, include_spans, max_len
            )
        if runtime == "constrained_classification":
            return self._run_classification(
                texts, schemas, sequences, include_confidence, max_len
            )
        if runtime == "joint_ie":
            return self._run_joint(
                texts, schemas, sequences, include_confidence, include_spans, max_len
            )
        raise ValueError(
            "'runtime' must be classic, constrained_classification, or joint_ie"
        )

    def _run_classic(
        self,
        texts: list[str],
        schemas: list[Any],
        sequences: list[torch.Tensor],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
        max_len: int | None,
    ) -> list[dict[str, Any]]:
        schema_mod = _require("gliner2.inference.schema", "GLiNER25 schema parse")
        parsed = [schema_mod.Schema.from_dict(schema) for schema in schemas]
        return self.extractor.batch_extract(
            texts,
            parsed,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
            hidden_states=sequences,
        )

    def _run_classification(
        self,
        texts: list[str],
        schemas: list[Any],
        sequences: list[torch.Tensor],
        include_confidence: bool,
        max_len: int | None,
    ) -> list[dict[str, Any]]:
        classification = _require("gliner2.classification", "GLiNER25 classifier")
        parsed = [classification.ClassificationSchema.from_dict(schema) for schema in schemas]
        config = classification.ClassificationConfig(
            include_confidence=include_confidence,
            max_len=max_len,
        )
        results = self.task_classifier.batch_classify(
            texts, parsed, config=config, hidden_states=sequences
        )
        return [_as_dict(result) for result in results]

    def _run_joint(
        self,
        texts: list[str],
        schemas: list[Any],
        sequences: list[torch.Tensor],
        include_confidence: bool,
        include_spans: bool,
        max_len: int | None,
    ) -> list[dict[str, Any]]:
        joint = _require("gliner2.joint_ie", "GLiNER25 joint IE")
        schema_mod = _require("gliner2.joint_ie.schema", "GLiNER25 joint schema")
        parsed = [schema_mod.JointSchema.from_dict(schema) for schema in schemas]
        config = joint.JointIEConfig(
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
        )
        results = self.joint_engine.batch_extract(
            texts, parsed, config=config, hidden_states=sequences
        )
        return [_as_dict(result) for result in results]


def _as_dict(result: Any) -> dict[str, Any]:
    """Return a JSON-ready dict from a GLiNER2 result.

    Args:
        result: A dict or an object with ``to_dict()``.

    Returns:
        The result mapping.

    Raises:
        TypeError: The result cannot be turned into a dict.
    """
    if isinstance(result, dict):
        return result
    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"{type(result).__name__} has no to_dict")
    converted = to_dict()
    if not isinstance(converted, dict):
        raise TypeError(f"{type(result).__name__}.to_dict() returned {type(converted).__name__}")
    return converted


def _pack_json(sample: dict[str, Any], device: torch.device) -> torch.Tensor:
    payload = json.dumps(sample, default=str).encode("utf-8")
    values = [float(len(payload)), *[float(byte) for byte in payload]]
    return torch.tensor(values, device=device, dtype=torch.float32)


def _empty_payloads(count: int, device: torch.device) -> list[torch.Tensor | None]:
    """``count`` decodable empty results, one per sequence in the batch."""
    return [_pack_json({}, device) for _ in range(count)]
