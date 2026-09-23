"""Construct the boundary pooler against a local GLiNER2 checkout.

The pooler is loaded by file path (see the ``pooler_mod`` fixture) so
poolers/__init__.py never imports ColBERT.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("gliner2", reason="boundary pooler construct needs gliner2")

_FIXTURES = Path(__file__).resolve().parents[3] / "GLiNER2" / "tests" / "fixtures"


def _load_fixture(name: str):
    spec = importlib.util.spec_from_file_location(f"gliner2_{name}", _FIXTURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TINY_HEAD = {
    "boundary_dim": 24,
    "pair_dim": 24,
    "start_top_k": 12,
    "end_top_k": 12,
    "ends_per_start": 6,
    "starts_per_end": 6,
    "candidate_budget": 48,
    "training_candidate_budget": 64,
    "max_gold_per_query": 16,
    "end_block_size": 16,
    "dropout": 0.0,
    "enable_records": True,
    "enable_relations": True,
}


def _build_pooler(pooler_mod: ModuleType):
    """Build an encoder-less pooler on the tiny offline fixture."""
    from gliner2.configuration import ExtractorConfig

    tokenizer = _load_fixture("tiny_tokenizer").build_tiny_tokenizer()
    encoder_config = _load_fixture("tiny_encoder").build_tiny_encoder_config(
        vocab_size=len(tokenizer), hidden_size=32
    )
    head = dict(_TINY_HEAD)
    config = ExtractorConfig(
        model_name="tiny-bert-fixture",
        architecture="boundary",
        boundary_head=head,
        token_pooling="first",
    )
    return pooler_mod.GLiNER25BoundaryPooler(
        config=config, encoder_config=encoder_config, tokenizer=tokenizer
    )


def test_boundary_pooler_constructs_and_exposes_checkpoint_prefixes(
    pooler_mod: ModuleType,
):
    pooler = _build_pooler(pooler_mod)
    assert pooler.extractor.encoder is None
    assert pooler.task_classifier.model is pooler.extractor
    assert pooler.joint_engine.model is pooler.extractor
    keys = set(pooler.extractor.state_dict().keys())
    assert any(key.startswith("classifier.") for key in keys)
    assert any(key.startswith("boundary_head.") for key in keys)
    assert any(key.startswith("record_decoder.") for key in keys)
    assert any(key.startswith("relation_scorer.") for key in keys)
    assert pooler.get_tasks() == {"embed", "classify", "plugin"}


def test_encoderless_model_refuses_to_encode_without_hidden_states(
    pooler_mod: ModuleType,
) -> None:
    pooler = _build_pooler(pooler_mod)

    with pytest.raises(ValueError, match="without an encoder"):
        pooler.extractor.encode_tokens(object(), hidden_states=None)
