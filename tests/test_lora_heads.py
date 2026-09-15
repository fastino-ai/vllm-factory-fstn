"""Unit tests for task-head LoRA conversion.

Requires vLLM (CPU is enough): conversion constructs ``ReplicatedLinear``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm", reason="head LoRA conversion requires vLLM")

try:
    from vllm.lora.utils import get_supported_lora_modules
    from vllm.model_executor.layers.linear import ReplicatedLinear
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"vLLM importable but layers unavailable ({exc!r})", allow_module_level=True)

from vllm_factory.pooling.lora_heads import (
    BOUNDARY_HEAD_NAMES,
    SPAN_HEAD_NAMES,
    convert_heads_to_replicated,
    head_weights_mapper,
    present_head_names,
)


@pytest.fixture(scope="module")
def vllm_parallel():
    """World-size-1 parallel group so ReplicatedLinear can construct."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:23457",
        local_rank=0,
        backend="gloo",
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield


class _FakeHeadTree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        self.span_rep = nn.Module()
        self.span_rep.out_project = nn.Sequential(nn.Linear(8, 8), nn.Identity(), nn.Linear(8, 8))
        self.count_pred = nn.Linear(8, 4)
        self.untouched = nn.Linear(8, 8)


def test_convert_replaces_only_named_heads_and_preserves_weights(vllm_parallel) -> None:
    tree = _FakeHeadTree()
    original = {
        "classifier.0": tree.classifier[0].weight.detach().clone(),
        "classifier.2": tree.classifier[2].weight.detach().clone(),
        "count_pred": tree.count_pred.weight.detach().clone(),
        "untouched": tree.untouched.weight.detach().clone(),
    }

    converted = convert_heads_to_replicated(
        tree, ("classifier", "span_rep", "count_pred", "count_embed")
    )

    assert converted == 5
    assert isinstance(tree.classifier[0], ReplicatedLinear)
    assert tree.classifier[0].return_bias is False
    assert isinstance(tree.untouched, nn.Linear)
    torch.testing.assert_close(tree.classifier[0].weight, original["classifier.0"])
    torch.testing.assert_close(tree.classifier[2].weight, original["classifier.2"])
    torch.testing.assert_close(tree.count_pred.weight, original["count_pred"])
    torch.testing.assert_close(tree.untouched.weight, original["untouched"])


def test_converted_heads_are_supported_lora_modules(vllm_parallel) -> None:
    tree = _FakeHeadTree()
    assert get_supported_lora_modules(tree) == []

    convert_heads_to_replicated(tree, ("classifier", "span_rep", "count_pred"))
    supported = set(get_supported_lora_modules(tree))
    assert supported == {"0", "2", "count_pred"}


def test_head_weights_mapper_rewrites_peft_prefixes() -> None:
    mapper = head_weights_mapper(("classifier", "span_rep"))
    assert mapper._map_name("classifier.0.lora_A.weight") == (
        "_business_pooler.classifier.0.lora_A.weight"
    )
    assert mapper._map_name("span_rep.span_rep_layer.out_project.0.lora_B.weight") == (
        "_business_pooler.span_rep.span_rep_layer.out_project.0.lora_B.weight"
    )
    assert mapper._map_name("encoder.layer.0.weight") == "encoder.layer.0.weight"


def test_present_head_names_skips_absent_optional_heads() -> None:
    tree = _FakeHeadTree()
    assert present_head_names(tree, SPAN_HEAD_NAMES) == (
        "span_rep",
        "classifier",
        "count_pred",
    )
    assert present_head_names(tree, BOUNDARY_HEAD_NAMES) == ("classifier",)
