"""SupportsLoRA protocol-compliance tests for the GLiNER 2.5 vLLM plugin."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="SupportsLoRA tests require vLLM")

try:
    from vllm.model_executor.models.interfaces import supports_lora  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"vLLM importable but interfaces unavailable ({exc!r})",
        allow_module_level=True,
    )


def test_gliner25_plugin_model_satisfies_supports_lora_protocol():
    from vllm.model_executor.models.interfaces import supports_lora

    from plugins.deberta_gliner25.model import GLiNER25VLLMModel

    assert supports_lora(GLiNER25VLLMModel)
    assert GLiNER25VLLMModel.supports_lora is True


def test_gliner25_plugin_model_prefixes_backbone_mapping():
    from plugins.deberta_gliner25.model import (
        _ENCODER_EMBEDDING_MODULES,
        _ENCODER_PACKED_MODULES_MAPPING,
        GLiNER25VLLMModel,
    )

    for key, val in _ENCODER_PACKED_MODULES_MAPPING.items():
        assert f"encoder.{key}" in GLiNER25VLLMModel.packed_modules_mapping
        assert GLiNER25VLLMModel.packed_modules_mapping[f"encoder.{key}"] == [
            f"encoder.{n}" for n in val
        ]

    for key, val in _ENCODER_EMBEDDING_MODULES.items():
        assert GLiNER25VLLMModel.embedding_modules[f"encoder.{key}"] == val

    expected_packed_keys = {f"encoder.{k}" for k in _ENCODER_PACKED_MODULES_MAPPING}
    expected_embed_keys = {f"encoder.{k}" for k in _ENCODER_EMBEDDING_MODULES}
    assert set(GLiNER25VLLMModel.packed_modules_mapping) == expected_packed_keys
    assert set(GLiNER25VLLMModel.embedding_modules) == expected_embed_keys


def test_gliner25_plugin_maps_task_heads_onto_the_pooler_path():
    from vllm_factory.pooling.lora_heads import BOUNDARY_HEAD_NAMES, head_weights_mapper

    mapper = head_weights_mapper(BOUNDARY_HEAD_NAMES)
    assert mapper._map_name("classifier.3.lora_A.weight") == (
        "_business_pooler.classifier.3.lora_A.weight"
    )
    assert mapper._map_name("boundary_head.pair_scorer.compat_mix.lora_B.weight") == (
        "_business_pooler.boundary_head.pair_scorer.compat_mix.lora_B.weight"
    )
    assert mapper._map_name("record_decoder.q_proj.weight") == (
        "_business_pooler.record_decoder.q_proj.weight"
    )
    assert BOUNDARY_HEAD_NAMES == (
        "classifier",
        "boundary_head",
        "record_decoder",
        "relation_scorer",
    )
