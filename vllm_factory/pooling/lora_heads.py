"""Make GLiNER task heads adapter-eligible under vLLM LoRA.

vLLM only collects ``LinearBase`` modules into ``expected_lora_modules``.
GLiNER heads are built from ``torch.nn.Linear``, so an adapter carrying
task-head weights is rejected at load. Converting those linears to
``ReplicatedLinear(return_bias=False)`` makes them eligible, and a
``WeightsMapper`` rewrites PEFT's top-level head prefixes onto the
``_business_pooler.`` path the serving model uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

POOLER_LORA_PREFIX = "_business_pooler."

SPAN_HEAD_NAMES: tuple[str, ...] = (
    "span_rep",
    "classifier",
    "count_pred",
    "count_embed",
)
BOUNDARY_HEAD_NAMES: tuple[str, ...] = (
    "classifier",
    "boundary_head",
    "record_decoder",
    "relation_scorer",
)


def present_head_names(root: nn.Module, candidates: Sequence[str]) -> tuple[str, ...]:
    """Return candidate head names that exist as attributes on ``root``.

    Args:
        root: Pooler holding the task heads.
        candidates: Head names to keep when present.

    Returns:
        The subset of ``candidates`` that resolve on ``root``.
    """
    return tuple(name for name in candidates if hasattr(root, name))


def convert_heads_to_replicated(root: nn.Module, head_names: Sequence[str]) -> int:
    """Replace ``nn.Linear`` under each named head with ``ReplicatedLinear``.

    ``return_bias=False`` keeps a tensor-returning drop-in for heads built as
    ``nn.Sequential``. ``disable_tp=True`` because these heads are not sharded.

    Args:
        root: Pooler whose named head subtrees should be converted.
        head_names: Attribute names on ``root`` to convert.

    Returns:
        Number of ``nn.Linear`` modules replaced.
    """
    converted = 0
    for name in head_names:
        head = getattr(root, name, None)
        if head is None:
            continue
        if isinstance(head, nn.Linear):
            _replace_linear(root, name, head)
            converted += 1
        else:
            converted += _convert_module(head)
    return converted


def head_weights_mapper(head_names: Sequence[str], prefix: str = POOLER_LORA_PREFIX):
    """Map PEFT head prefixes onto the serving model's pooler path.

    Args:
        head_names: Task-head attribute names as they appear in the adapter.
        prefix: Module path of the pooler on the vLLM model.

    Returns:
        A ``WeightsMapper`` that rewrites ``<head>.`` to ``<prefix><head>.``.
    """
    from vllm.model_executor.models.utils import WeightsMapper

    return WeightsMapper(orig_to_new_prefix={f"{name}.": f"{prefix}{name}." for name in head_names})


def _convert_module(module: nn.Module) -> int:
    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            _replace_linear(module, name, child)
            converted += 1
        else:
            converted += _convert_module(child)
    return converted


def _replace_linear(parent: nn.Module, name: str, child: nn.Linear) -> None:
    from vllm.model_executor.layers.linear import ReplicatedLinear

    replacement = ReplicatedLinear(
        child.in_features,
        child.out_features,
        bias=child.bias is not None,
        params_dtype=child.weight.dtype,
        return_bias=False,
        disable_tp=True,
    )
    with torch.no_grad():
        replacement.weight.copy_(child.weight)
        if child.bias is not None:
            replacement.bias.copy_(child.bias)
    parent._modules[name] = replacement
