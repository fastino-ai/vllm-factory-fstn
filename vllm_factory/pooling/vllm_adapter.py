"""vLLM pooler adapter — the ONE file that touches vLLM pooler internals.

When vLLM changes its ``Pooler`` ABC or ``PoolingMetadata`` layout, update
THIS file.  Business poolers (GLiNER, ColBERT, ...) and model files stay
untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Set
from typing import Any

import torch

# ── vLLM imports (the ONLY place in the project that touches these) ──────────
from vllm.config import PoolerConfig
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.v1.pool.metadata import PoolingMetadata

from vllm_factory.lora.routing import batch_routing, uniform_scope
from vllm_factory.pooling.protocol import (
    NO_ADAPTER,
    FactoryPooler,
    PassthroughPooler,
    PoolerContext,
    null_lora_scope,
)

logger = logging.getLogger(__name__)

PoolerOutput = list[torch.Tensor | None]


def _lora_binding(count: int) -> tuple[tuple[int, ...], Any]:
    """Build the per-sequence adapter slots and scope factory for this batch.

    The model's forward pass publishes the routing it resolved for the same
    batch, so the pooler reuses it rather than re-reading vLLM's step mapping.

    Args:
        count: Number of scheduled sequences.

    Returns:
        A ``(slots, scope)`` pair; ``scope`` takes a sequence index and returns
        a context manager binding head projections to that sequence's adapter.
    """
    routing = batch_routing()
    if routing is None or routing.wrapper is None or len(routing.slots) != count:
        return (NO_ADAPTER,) * count, null_lora_scope

    def scope(seq_index: int):
        return uniform_scope(routing.wrapper, routing.slots[seq_index])

    return routing.slots, scope


def _translate_metadata(pm: PoolingMetadata) -> PoolerContext:
    """Convert vLLM's ``PoolingMetadata`` into the stable ``PoolerContext``."""
    seq_lengths = pm.prompt_lens.tolist()
    prompt_token_ids: list[torch.Tensor] = []
    try:
        prompt_token_ids = pm.get_prompt_token_ids()
    except (AssertionError, AttributeError):
        prompt_token_ids = []
    extra_kwargs: list[dict[str, Any]] = []
    tasks: list[str] = []
    for pp in pm.pooling_params:
        ek = getattr(pp, "extra_kwargs", None) or {}
        extra_kwargs.append(ek)
        tasks.append(pp.task or "plugin")
    lora_slots, lora_scope = _lora_binding(len(seq_lengths))
    return PoolerContext(
        seq_lengths=seq_lengths,
        extra_kwargs=extra_kwargs,
        tasks=tasks,
        prompt_token_ids=prompt_token_ids,
        lora_slots=lora_slots,
        lora_scope=lora_scope,
    )


class VllmPoolerAdapter(Pooler):
    """Wraps any :class:`FactoryPooler` into vLLM's ``Pooler`` interface.

    For :class:`PassthroughPooler` instances (models that handle projection
    in their ``forward()``), delegates to vLLM's native ``pooler_for_token_embed``
    for maximum throughput and scheduler integration.

    For custom business poolers (GLiNER, etc.), translates
    ``PoolingMetadata`` into the stable ``PoolerContext`` and delegates
    to the inner pooler's ``forward()``.

    Args:
        factory_pooler: Business pooling logic (GLiNER, ColBERT, ...).
        pooler_config: vLLM PoolerConfig — required for PassthroughPooler to
            create the native TokenPooler.  Falls back to ``PoolerConfig(pooling_type="ALL")``.
        requires_token_ids: Whether to request prompt token IDs from the
            scheduler (needed by GLiNER / linker poolers).
    """

    def __init__(
        self,
        factory_pooler: FactoryPooler,
        *,
        pooler_config: PoolerConfig | None = None,
        requires_token_ids: bool = False,
    ) -> None:
        super().__init__()
        self._inner = factory_pooler
        self._requires_token_ids = requires_token_ids
        self._use_native = isinstance(factory_pooler, PassthroughPooler)

        if self._use_native:
            if pooler_config is None:
                pooler_config = PoolerConfig(pooling_type="ALL")
            self._native = pooler_for_token_embed(
                pooler_config,
                projector=lambda x: x,
            )

    # ── Pooler ABC ────────────────────────────────────────────────────────

    def get_supported_tasks(self) -> Set:
        if self._use_native:
            return self._native.get_supported_tasks()
        return self._inner.get_tasks()

    def get_pooling_updates(self, task) -> PoolingParamsUpdate:
        if self._use_native:
            return self._native.get_pooling_updates(task)
        return PoolingParamsUpdate(requires_token_ids=self._requires_token_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        if self._use_native:
            return self._native(hidden_states, pooling_metadata)
        ctx = _translate_metadata(pooling_metadata)
        return self._inner.forward(hidden_states, ctx)
