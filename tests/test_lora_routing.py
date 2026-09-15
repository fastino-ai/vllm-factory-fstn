"""Unit tests for per-row Punica LoRA routing.

The kernels index adapter weights by activation row, so the metadata a scope
installs has to describe exactly the rows a layer projects. Every test here
checks that against what vLLM's own ``prepare_tensors`` would have produced.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm", reason="LoRA routing builds vLLM kernel metadata")

try:
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"vLLM importable but LoRA ops unavailable ({exc!r})", allow_module_level=True)

from vllm_factory.lora.routing import (
    _SCRATCH,
    NO_LORA,
    BatchRouting,
    _RowMapping,
    clear_batch_routing,
    encoder_scope,
    find_punica_wrapper,
    max_distinct_slots,
    observed_slot_sets,
    project_shared_table,
    reset_observed_slots,
    row_scope,
    sequence_slots,
    set_batch_routing,
    uniform_scope,
)
from vllm_factory.pooling.protocol import PoolerContext

MAX_LORAS = 4
CAPACITY = 4096

META_FIELDS = (
    "token_lora_mapping",
    "token_indices_sorted_by_lora_ids",
    "active_lora_ids",
    "num_tokens_per_lora",
    "lora_token_start_loc",
    "no_lora_flag_cpu",
    "num_active_loras_cpu",
)
ROW_FIELDS = ("token_lora_mapping", "token_indices_sorted_by_lora_ids")


class _FakeWrapper:
    """Stands in for PunicaWrapperGPU: the state the scopes read and swap."""

    def __init__(self, token_slots: list[int] | None = None) -> None:
        self.max_loras = MAX_LORAS
        self.device = torch.device("cpu")
        self.token_mapping_meta = LoRAKernelMeta.make(MAX_LORAS, CAPACITY, device="cpu")
        self.indices_len: list[int | None] = [None] * 4
        self._token_lora_indices = torch.empty(0, dtype=torch.long)
        if token_slots is not None:
            self._token_lora_indices = torch.tensor(token_slots, dtype=torch.long)
            self.indices_len[0] = len(token_slots)

    @property
    def token_lora_indices(self) -> torch.Tensor:
        return self._token_lora_indices[: self.indices_len[0]]


def _reference(row_slots: list[int]) -> LoRAKernelMeta:
    meta = LoRAKernelMeta.make(MAX_LORAS, CAPACITY, device="cpu")
    meta.prepare_tensors(torch.tensor(row_slots, dtype=torch.int32))
    return meta


def _assert_meta_matches(got: tuple, row_slots: list[int]) -> None:
    """Compare an installed mapping's kernel args against vLLM's own metadata."""
    ref = _reference(row_slots)
    rows = len(row_slots)
    expected = ref.meta_args(rows, specialize_active_lora=True)
    no_lora = bool(ref.no_lora_flag_cpu[0])

    for name, want, have in zip(META_FIELDS, expected, got, strict=True):
        if no_lora and name in ROW_FIELDS:
            continue  # vLLM leaves both buffers uninitialised on the no-lora path.
        assert torch.equal(want, have), f"{name}: want {want.tolist()}, got {have.tolist()}"


def _installed_args(wrapper: _FakeWrapper, rows: int) -> tuple:
    return wrapper.token_mapping_meta.meta_args(rows, specialize_active_lora=True)


@pytest.fixture(autouse=True)
def _clean_routing_state():
    reset_observed_slots()
    clear_batch_routing()
    yield
    reset_observed_slots()
    clear_batch_routing()


@pytest.mark.parametrize(
    "row_slots",
    [
        [0] * 5 + [1] * 3,
        [1, 0, 1, 0, 1],
        [0, -1, 0, 1],
        [-1, -1, -1],
        [2] * 9,
        [0, 1, 2, 3],
    ],
)
def test_row_scope_metadata_matches_vllm(row_slots: list[int]) -> None:
    wrapper = _FakeWrapper()
    with row_scope(wrapper, row_slots):
        _assert_meta_matches(_installed_args(wrapper, len(row_slots)), row_slots)


@pytest.mark.parametrize("slot", [0, 1, 3, NO_LORA])
@pytest.mark.parametrize("rows", [1, 7, 300, 1024, 2500])
def test_uniform_scope_metadata_matches_vllm(slot: int, rows: int) -> None:
    wrapper = _FakeWrapper()
    with uniform_scope(wrapper, slot):
        _assert_meta_matches(_installed_args(wrapper, rows), [slot] * rows)


def test_uniform_scope_resizes_per_layer_within_one_scope() -> None:
    """A head runs span, label and count shapes under one adapter."""
    wrapper = _FakeWrapper()
    with uniform_scope(wrapper, 2):
        for rows in (1140, 6, 1, 1140, 3000):
            _assert_meta_matches(_installed_args(wrapper, rows), [2] * rows)


def test_scopes_restore_the_step_mapping() -> None:
    wrapper = _FakeWrapper()
    original = wrapper.token_mapping_meta
    with uniform_scope(wrapper, 1):
        assert wrapper.token_mapping_meta is not original
        with row_scope(wrapper, [0, 1]):
            assert wrapper.token_mapping_meta is not original
        assert wrapper.token_mapping_meta is not original
    assert wrapper.token_mapping_meta is original


def test_nested_row_scopes_do_not_share_a_buffer() -> None:
    """The rel-pos table nests inside the encoder scope and can round to its capacity."""
    wrapper = _FakeWrapper()
    outer = [0] * 80 + [1] * 80
    inner = [0] * 512 + [1] * 512

    with row_scope(wrapper, outer):
        _assert_meta_matches(_installed_args(wrapper, len(outer)), outer)
        with row_scope(wrapper, inner):
            _assert_meta_matches(_installed_args(wrapper, len(inner)), inner)
        _assert_meta_matches(_installed_args(wrapper, len(outer)), outer)


def test_nested_row_scope_buffers_are_reused_across_forwards() -> None:
    """A buffer per nesting level, not per call, or serving would leak memory."""
    wrapper = _FakeWrapper()
    for _ in range(3):
        with row_scope(wrapper, [0, 1]), row_scope(wrapper, [1, 0]):
            pass
    assert [len(pool) for pool in _SCRATCH[wrapper].values()] == [2]


def test_row_scope_rejects_a_row_count_it_cannot_describe() -> None:
    """Silently mis-sizing the mapping would read adapter weights out of bounds."""
    wrapper = _FakeWrapper()
    with row_scope(wrapper, [0, 0, 1, 1]), pytest.raises(ValueError, match="describes 4 rows"):
        _installed_args(wrapper, 8)


def test_row_scope_metadata_survives_a_capacity_bucket_change() -> None:
    wrapper = _FakeWrapper()
    for rows in (10, 1500, 10):
        slots = [0] * rows
        with row_scope(wrapper, slots):
            _assert_meta_matches(_installed_args(wrapper, rows), slots)


def test_no_wrapper_leaves_every_scope_inert() -> None:
    with uniform_scope(None, 0), row_scope(None, [0]), encoder_scope(None, [0], 4):
        pass


def test_sequence_slots_reads_the_slot_at_each_sequence_start() -> None:
    wrapper = _FakeWrapper([7, 7, 7, 3, 3, 9])
    assert sequence_slots(wrapper, [3, 2, 1]) == [7, 3, 9]


def test_sequence_slots_falls_back_when_the_step_mapping_is_absent() -> None:
    assert sequence_slots(_FakeWrapper(), [3, 2]) == [NO_LORA, NO_LORA]
    assert sequence_slots(None, [3, 2]) == [NO_LORA, NO_LORA]


def test_sequence_slots_falls_back_when_the_mapping_is_short() -> None:
    wrapper = _FakeWrapper([1, 1])
    assert sequence_slots(wrapper, [3, 2]) == [NO_LORA, NO_LORA]


def test_sequence_slots_records_the_adapters_seen_per_forward() -> None:
    sequence_slots(_FakeWrapper([0, 0, 1]), [2, 1])
    sequence_slots(_FakeWrapper([5, 5]), [2])
    assert observed_slot_sets() == [frozenset({0, 1}), frozenset({5})]
    assert max_distinct_slots() == 2


def test_encoder_scope_maps_padded_rows_to_their_sequence() -> None:
    wrapper = _FakeWrapper()
    with encoder_scope(wrapper, [0, 1, 0], width=4):
        _assert_meta_matches(_installed_args(wrapper, 12), [0] * 4 + [1] * 4 + [0] * 4)


def test_encoder_scope_on_one_adapter_sizes_itself_to_the_layer() -> None:
    wrapper = _FakeWrapper()
    with encoder_scope(wrapper, [2, 2], width=5):
        _assert_meta_matches(_installed_args(wrapper, 10), [2] * 10)


class _RecordingLinear(nn.Module):
    """Linear that records the rows it was asked to project, as Punica sees them."""

    def __init__(self, wrapper: _FakeWrapper, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4, out_features)
        self._wrapper = wrapper
        self.seen: list[tuple] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        rows = int(x.shape[0] * x.shape[1])
        self.seen.append(_installed_args(self._wrapper, rows))
        return self.linear(x), None


def test_project_shared_table_keeps_one_row_for_a_single_adapter() -> None:
    wrapper = _FakeWrapper()
    layer = _RecordingLinear(wrapper, 6)
    table = torch.zeros(1, 8, 4)
    set_batch_routing(BatchRouting(wrapper=wrapper, slots=(1, 1, 1)))

    out = project_shared_table(layer, table, batch_rows=3)

    assert out.shape == (1, 8, 6)
    _assert_meta_matches(layer.seen[0], [1] * 8)


def test_project_shared_table_expands_per_sequence_for_mixed_adapters() -> None:
    """Each sequence's position bias has to come from its own adapter."""
    wrapper = _FakeWrapper()
    layer = _RecordingLinear(wrapper, 6)
    table = torch.zeros(1, 8, 4)
    set_batch_routing(BatchRouting(wrapper=wrapper, slots=(0, 2, 0)))

    out = project_shared_table(layer, table, batch_rows=3)

    assert out.shape == (3, 8, 6)
    _assert_meta_matches(layer.seen[0], [0] * 8 + [2] * 8 + [0] * 8)


def test_project_shared_table_ignores_routing_from_a_different_batch() -> None:
    wrapper = _FakeWrapper()
    layer = _RecordingLinear(wrapper, 6)
    set_batch_routing(BatchRouting(wrapper=wrapper, slots=(0, 1)))

    out = project_shared_table(layer, torch.zeros(1, 8, 4), batch_rows=3)

    assert out.shape == (1, 8, 6)


def test_find_punica_wrapper_returns_the_shared_wrapper() -> None:
    wrapper = _FakeWrapper()
    model = nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2)))
    assert find_punica_wrapper(model) is None

    model[1][0].punica_wrapper = wrapper
    assert find_punica_wrapper(model) is wrapper


def test_pooler_context_reports_whether_one_adapter_covers_the_batch() -> None:
    assert PoolerContext(seq_lengths=[1, 2], lora_slots=(3, 3)).shares_one_adapter()
    assert PoolerContext(seq_lengths=[1, 2]).shares_one_adapter()
    assert not PoolerContext(seq_lengths=[1, 2], lora_slots=(3, 4)).shares_one_adapter()


def test_row_mapping_passes_a_matching_row_count_through() -> None:
    meta = _reference([0, 1])
    mapping = _RowMapping(meta, 2, [])
    assert mapping.meta_args(2, True)[0].tolist() == [0, 1]
