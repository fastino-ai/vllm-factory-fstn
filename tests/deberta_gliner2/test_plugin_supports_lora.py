"""SupportsLoRA protocol-compliance tests for the GLiNER2 vLLM plugin.

Verifies that the plugin's top-level model class — the one vLLM sees after
``register_plugin("gliner2", ...)`` — satisfies ``SupportsLoRA`` and
correctly prefixes the underlying DeBERTa v2/v3 backbone's LoRA metadata
with ``encoder.`` so vLLM's LoRA manager walks through ``self.encoder``
when resolving adapter target modules.

Requires vLLM to be importable (CPU-only environments skip).
"""

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


def test_gliner2_plugin_model_satisfies_supports_lora_protocol():
    from vllm.model_executor.models.interfaces import supports_lora

    from plugins.deberta_gliner2.model import GLiNER2VLLMModel

    assert supports_lora(GLiNER2VLLMModel)
    assert GLiNER2VLLMModel.supports_lora is True


def test_gliner2_plugin_model_prefixes_backbone_mapping():
    """Plugin model must re-export the backbone's LoRA metadata under the
    ``encoder.`` prefix — that's where the backbone lives in the module
    tree (``GLiNER2VLLMModel.encoder``)."""
    from plugins.deberta_gliner2.model import (
        _ENCODER_EMBEDDING_MODULES,
        _ENCODER_PACKED_MODULES_MAPPING,
        GLiNER2VLLMModel,
    )

    # Every backbone key must reappear under ``encoder.<key>`` in the
    # plugin-level mapping, and each constituent name in the value list
    # must likewise be prefixed.
    for key, val in _ENCODER_PACKED_MODULES_MAPPING.items():
        assert f"encoder.{key}" in GLiNER2VLLMModel.packed_modules_mapping
        assert GLiNER2VLLMModel.packed_modules_mapping[f"encoder.{key}"] == [
            f"encoder.{n}" for n in val
        ]

    for key, val in _ENCODER_EMBEDDING_MODULES.items():
        assert GLiNER2VLLMModel.embedding_modules[f"encoder.{key}"] == val

    # Heads are adapter-eligible via ReplicatedLinear conversion; they are not
    # packed, so packed_modules_mapping stays encoder-only.
    expected_packed_keys = {f"encoder.{k}" for k in _ENCODER_PACKED_MODULES_MAPPING}
    expected_embed_keys = {f"encoder.{k}" for k in _ENCODER_EMBEDDING_MODULES}
    assert set(GLiNER2VLLMModel.packed_modules_mapping) == expected_packed_keys
    assert set(GLiNER2VLLMModel.embedding_modules) == expected_embed_keys


def test_gliner2_plugin_maps_task_heads_onto_the_pooler_path():
    from plugins.deberta_gliner2.model import GLiNER2VLLMModel
    from vllm_factory.pooling.lora_heads import SPAN_HEAD_NAMES, head_weights_mapper

    mapper = head_weights_mapper(SPAN_HEAD_NAMES)
    assert mapper._map_name("classifier.0.lora_A.weight") == (
        "_business_pooler.classifier.0.lora_A.weight"
    )
    assert mapper._map_name("span_rep.span_rep_layer.out_project.0.weight") == (
        "_business_pooler.span_rep.span_rep_layer.out_project.0.weight"
    )
    # Class-level packed mapping is unchanged; the instance mapper is what
    # LoRA loading consults, and SPAN_HEAD_NAMES is the conversion surface.
    assert SPAN_HEAD_NAMES == ("span_rep", "classifier", "count_pred", "count_embed")
    assert GLiNER2VLLMModel.supports_lora is True
