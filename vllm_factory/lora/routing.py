from __future__ import annotations

import bisect
import contextlib
import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

NO_LORA = -1

_WRAPPERS: weakref.WeakKeyDictionary[nn.Module, Any] = weakref.WeakKeyDictionary()
_SCRATCH: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], list[Any]]] = (
    weakref.WeakKeyDictionary()
)
_DEPTH: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], int]] = weakref.WeakKeyDictionary()
_OBSERVED: list[frozenset[int]] = []
_CURRENT: "BatchRouting | None" = None

_CAPACITY_STEP = 1024


@dataclass(frozen=True)
class BatchRouting:
    """Adapter routing for the batch currently in the model's forward pass.

    Attributes:
        wrapper: The shared Punica wrapper every LoRA layer of the model holds.
        slots: Adapter slot per scheduled sequence, ``NO_LORA`` where none
            applies, in the order the runner scheduled them.
    """

    wrapper: Any
    slots: tuple[int, ...]


def find_punica_wrapper(model: nn.Module) -> Any | None:
    """Return the Punica wrapper shared by the model's LoRA layers.

    vLLM's LoRA manager hands one wrapper by reference to every layer it
    wraps, so the first one found speaks for the whole model. The manager runs
    after the model is built, so this resolves on first use, not at init.

    Args:
        model: Root module of the serving model.

    Returns:
        The wrapper, or ``None`` when LoRA is disabled or no layer was wrapped.
    """
    cached = _WRAPPERS.get(model)
    if cached is not None:
        return cached
    for module in model.modules():
        wrapper = getattr(module, "punica_wrapper", None)
        if wrapper is not None:
            _WRAPPERS[model] = wrapper
            return wrapper
    return None


def sequence_slots(wrapper: Any, seq_lengths: Sequence[int]) -> list[int]:
    """Read the adapter slot each scheduled sequence is bound to.

    The runner already resolved every token to a slot for this step, so the
    value at a sequence's first token is that sequence's slot. Records the
    result for :func:`observed_slot_sets`.

    Args:
        wrapper: Punica wrapper for the model.
        seq_lengths: Token count per scheduled sequence, in batch order.

    Returns:
        One slot per sequence, ``NO_LORA`` where the sequence carries no
        adapter or the step mapping is unavailable.
    """
    blank = [NO_LORA] * len(seq_lengths)
    if wrapper is None or wrapper.indices_len[0] is None:
        _OBSERVED.append(frozenset())
        return blank

    total = int(sum(seq_lengths))
    indices = wrapper.token_lora_indices
    if total == 0 or indices.numel() < total:
        _OBSERVED.append(frozenset())
        return blank

    values = indices[:total].tolist()
    slots: list[int] = []
    offset = 0
    for length in seq_lengths:
        slots.append(int(values[offset]) if length > 0 else NO_LORA)
        offset += length
    _OBSERVED.append(frozenset(slot for slot in slots if slot != NO_LORA))
    return slots


@contextlib.contextmanager
def row_scope(wrapper: Any, row_slots: Sequence[int]) -> Iterator[None]:
    """Install metadata mapping each activation row to its adapter slot.

    Every layer reached must project exactly ``len(row_slots)`` rows, because
    the mapping names a slot per row and nothing else describes the rest.

    Args:
        wrapper: Punica wrapper for the model.
        row_slots: Adapter slot per row of the activation about to be
            projected, in row order.

    Yields:
        Control, with the mapping active for every LoRA layer reached.
    """
    if wrapper is None:
        yield
        return
    captured = list(wrapper.token_mapping_meta.captured_lora_counts)
    key = ("rows", _capacity(len(row_slots)))
    meta = _checkout(wrapper, key, captured)
    try:
        _fill_rows(meta, row_slots, wrapper.max_loras)
        with _installed(wrapper, _RowMapping(meta, len(row_slots), captured)):
            yield
    finally:
        _checkin(wrapper, key)


@contextlib.contextmanager
def encoder_scope(wrapper: Any, slots: Sequence[int], width: int) -> Iterator[None]:
    """Install the row mapping for a batch padded to one row per sequence.

    Args:
        wrapper: Punica wrapper for the model.
        slots: Adapter slot per sequence, in batch order.
        width: Padded token width every sequence row was scattered into.

    Yields:
        Control, with the mapping active for every encoder projection.
    """
    if wrapper is None or not slots:
        yield
        return
    if len(set(slots)) == 1:
        with uniform_scope(wrapper, slots[0]):
            yield
        return
    with row_scope(wrapper, [slot for slot in slots for _ in range(width)]):
        yield


@contextlib.contextmanager
def uniform_scope(wrapper: Any, slot: int) -> Iterator[None]:
    """Install metadata binding every activation row to one adapter slot.

    The grouping holds whatever a layer projects, so the row count is taken
    from the layer instead of being fixed up front — a pooler head runs many
    shapes under one adapter. Knowing the grouping also skips the sort and the
    device-to-host read that :meth:`LoRAKernelMeta.prepare_tensors` performs.

    Args:
        wrapper: Punica wrapper for the model.
        slot: Adapter slot every row belongs to, or ``NO_LORA`` for none.

    Yields:
        Control, with the mapping active for every LoRA layer reached.
    """
    if wrapper is None:
        yield
        return
    captured = list(wrapper.token_mapping_meta.captured_lora_counts)
    with _installed(wrapper, _UniformMapping(wrapper, slot, captured)):
        yield


def project_shared_table(layer: nn.Module, table: torch.Tensor, batch_rows: int) -> torch.Tensor:
    """Project a batch-shared table through a possibly LoRA-wrapped linear.

    The relative-position table is one tensor for the whole batch, but each
    sequence may carry a different adapter, and PEFT applies the projection's
    adapter here too. One adapter keeps the shared single-row projection; more
    than one expands the table so every sequence gets its own rows.

    Args:
        layer: Linear module the table is projected through.
        table: Shared table of shape (1, entries, hidden).
        batch_rows: Number of sequence rows the caller will consume.

    Returns:
        The projection, shaped (1, entries, out) when one adapter covers the
        batch and (batch_rows, entries, out) when several do.
    """
    routing = _CURRENT
    wrapper = routing.wrapper if routing else None
    slots = routing.slots if routing else ()
    entries = int(table.shape[-2])

    if wrapper is None or len(slots) != batch_rows:
        return _unwrap(layer(table))
    if len(set(slots)) == 1:
        with uniform_scope(wrapper, slots[0]):
            return _unwrap(layer(table))

    rows = [slot for slot in slots for _ in range(entries)]
    with row_scope(wrapper, rows):
        return _unwrap(layer(table.expand(batch_rows, -1, -1)))


def observed_slot_sets() -> list[frozenset[int]]:
    """Return the distinct adapter slots seen in each recorded forward pass."""
    return list(_OBSERVED)


def max_distinct_slots() -> int:
    """Return the largest number of adapters routed within a single forward."""
    return max((len(seen) for seen in _OBSERVED), default=0)


def reset_observed_slots() -> None:
    """Drop the recorded per-forward slot sets."""
    _OBSERVED.clear()


def set_batch_routing(routing: BatchRouting | None) -> None:
    """Publish the routing the pooler should reuse for the current batch."""
    global _CURRENT
    _CURRENT = routing


def batch_routing() -> BatchRouting | None:
    """Return the routing published by the model's forward pass, if any."""
    return _CURRENT


def clear_batch_routing() -> None:
    """Forget the published routing so a later batch cannot inherit it."""
    global _CURRENT
    _CURRENT = None


class _RowMapping:
    """Metadata for rows whose adapter slots were resolved ahead of the call."""

    def __init__(self, meta: Any, rows: int, captured: list[int]) -> None:
        self._meta = meta
        self._rows = rows
        self.captured_lora_counts = captured

    def meta_args(self, token_nums: int, specialize_active_lora: bool) -> tuple[Any, ...]:
        """Return kernel metadata, rejecting a row count the mapping cannot describe.

        Args:
            token_nums: Rows the layer is about to project.
            specialize_active_lora: Whether to report the true active count.

        Returns:
            The metadata tuple Punica's shrink and expand kernels unpack.

        Raises:
            ValueError: When the layer projects a different number of rows
                than the mapping names slots for.
        """
        if token_nums != self._rows:
            raise ValueError(
                f"row_scope describes {self._rows} rows but the layer projected {token_nums}"
            )
        return self._meta.meta_args(token_nums, specialize_active_lora)


class _UniformMapping:
    """Metadata for rows that all share one adapter slot, sized at call time."""

    def __init__(self, wrapper: Any, slot: int, captured: list[int]) -> None:
        self._wrapper = wrapper
        self._slot = slot
        self.captured_lora_counts = captured

    def meta_args(self, token_nums: int, specialize_active_lora: bool) -> tuple[Any, ...]:
        """Return kernel metadata covering however many rows the layer projects.

        Args:
            token_nums: Rows the layer is about to project.
            specialize_active_lora: Whether to report the true active count.

        Returns:
            The metadata tuple Punica's shrink and expand kernels unpack.
        """
        meta = _uniform_meta(self._wrapper, self._slot, token_nums, self.captured_lora_counts)
        if self._slot != NO_LORA:
            meta.num_tokens_per_lora[0] = token_nums
            meta.lora_token_start_loc[1] = token_nums
        return meta.meta_args(token_nums, specialize_active_lora)


def _uniform_meta(wrapper: Any, slot: int, rows: int, captured: list[int]) -> Any:
    capacity = _capacity(rows)
    meta = _scratch_meta(wrapper, ("uniform", capacity, slot), captured)
    if getattr(meta, "_factory_ready", False):
        return meta

    meta._reset()
    if slot == NO_LORA:
        meta.no_lora_flag_cpu[0] = True
    else:
        meta.token_lora_mapping.fill_(slot)
        meta.token_indices_sorted_by_lora_ids.copy_(
            torch.arange(capacity, dtype=torch.int32), non_blocking=True
        )
        meta.active_lora_ids[0] = slot
        meta.num_active_loras_cpu[0] = _num_active(captured, 1)
    meta._factory_ready = True
    return meta


def _capacity(rows: int) -> int:
    return -(-max(rows, 1) // _CAPACITY_STEP) * _CAPACITY_STEP


def _unwrap(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


@contextlib.contextmanager
def _installed(wrapper: Any, meta: Any) -> Iterator[None]:
    previous = wrapper.token_mapping_meta
    wrapper.token_mapping_meta = meta
    try:
        yield
    finally:
        wrapper.token_mapping_meta = previous


def _make_meta(wrapper: Any, capacity: int, captured: list[int]) -> Any:
    from vllm.lora.ops.triton_ops import LoRAKernelMeta

    return LoRAKernelMeta.make(
        wrapper.max_loras,
        capacity,
        device=wrapper.device,
        captured_lora_counts=captured,
    )


def _scratch_meta(wrapper: Any, key: tuple[Any, ...], captured: list[int]) -> Any:
    by_key = _SCRATCH.setdefault(wrapper, {})
    pool = by_key.setdefault(key, [])
    if not pool:
        pool.append(_make_meta(wrapper, key[1], captured))
    return pool[0]


def _checkout(wrapper: Any, key: tuple[Any, ...], captured: list[int]) -> Any:
    """Lend a scratch buffer no enclosing scope is still using.

    The rel-pos table and the padded encoder tokens can round to the same
    capacity, and the inner scope runs inside the outer one, so handing both
    the same buffer would leave the outer scope describing the inner's rows.

    Args:
        wrapper: Punica wrapper for the model.
        key: Cache key naming the layout and its capacity.
        captured: Active-LoRA counts the kernels were captured for.

    Returns:
        A buffer reserved until the matching :func:`_checkin`.
    """
    pool = _SCRATCH.setdefault(wrapper, {}).setdefault(key, [])
    depths = _DEPTH.setdefault(wrapper, {})
    depth = depths.get(key, 0)
    while len(pool) <= depth:
        pool.append(_make_meta(wrapper, key[1], captured))
    depths[key] = depth + 1
    return pool[depth]


def _checkin(wrapper: Any, key: tuple[Any, ...]) -> None:
    _DEPTH[wrapper][key] -= 1


def _fill_rows(meta: Any, row_slots: Sequence[int], max_loras: int) -> None:
    mapping = torch.tensor(list(row_slots), dtype=torch.int32)
    rows = int(mapping.numel())
    meta._reset()
    if rows == 0 or bool(torch.all(mapping == NO_LORA)):
        meta.no_lora_flag_cpu[0] = True
        return

    order = torch.argsort(mapping, stable=True).to(torch.int32)
    ids, counts = torch.unique(mapping, sorted=True, return_counts=True)
    counts = counts.to(torch.int32)
    active = int(ids.numel())

    meta.token_lora_mapping[:rows].copy_(mapping, non_blocking=True)
    meta.token_indices_sorted_by_lora_ids[:rows].copy_(order, non_blocking=True)
    meta.active_lora_ids[:active].copy_(ids.to(torch.int32), non_blocking=True)
    meta.num_tokens_per_lora[:active].copy_(counts, non_blocking=True)
    meta.lora_token_start_loc[1 : 1 + active].copy_(
        torch.cumsum(counts, dim=0).to(torch.int32), non_blocking=True
    )
    meta.num_active_loras_cpu[0] = _num_active(meta.captured_lora_counts, active)


def _num_active(captured: Sequence[int], count: int) -> int:
    if captured and count > 0:
        index = bisect.bisect_left(captured, count)
        if index < len(captured):
            return captured[index]
    return count
