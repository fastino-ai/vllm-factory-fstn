"""Schema parse for each hosted runtime, valid and invalid."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from plugins.deberta_gliner25.processor import prompt_schema

pytest.importorskip("gliner2", reason="runtime parse needs gliner2")

_PARITY = (
    Path(__file__).resolve().parents[2] / "plugins" / "deberta_gliner25" / "parity_test.py"
)
_spec = importlib.util.spec_from_file_location("gliner25_parity_cases", str(_PARITY))
_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parity)
CASES = _parity.CASES


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_ticket_payload_parses(case: dict):
    prompt = prompt_schema(case["runtime"], case["schema"])
    assert isinstance(prompt, dict)
    assert prompt


def test_unknown_runtime_is_rejected():
    with pytest.raises(ValueError, match="runtime"):
        prompt_schema("span", {"entities": ["person"]})


def test_classic_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        prompt_schema("classic", {"entities": ["person"], "nope": True})


def test_classification_without_labels_is_rejected():
    with pytest.raises(ValueError):
        prompt_schema(
            "constrained_classification",
            {"tasks": {"intent": {}}},
        )


def test_joint_relation_to_an_unknown_entity_is_rejected():
    with pytest.raises(ValueError, match="unknown entity"):
        prompt_schema(
            "joint_ie",
            {
                "entities": ["person"],
                "relations": {"works_for": {"head": "person", "tail": "organization"}},
            },
        )


def test_schema_must_be_an_object():
    with pytest.raises(ValueError, match="object"):
        prompt_schema("classic", ["person"])
