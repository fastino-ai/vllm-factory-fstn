"""The pooler's output list must stay paired with the scheduled batch.

A coalesced batch holds several callers, so returning the wrong number of
payloads hands one caller another's extraction.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import torch

_PROTOCOL = (
    Path(__file__).resolve().parents[2] / "vllm_factory" / "pooling" / "protocol.py"
)
_protocol_spec = importlib.util.spec_from_file_location("gliner25_pooler_protocol", _PROTOCOL)
_protocol = importlib.util.module_from_spec(_protocol_spec)
sys.modules["gliner25_pooler_protocol"] = _protocol
_protocol_spec.loader.exec_module(_protocol)
PoolerContext = _protocol.PoolerContext


def _decode(payload: torch.Tensor) -> dict:
    """Unpack a payload the way the IO processor does."""
    data = payload.tolist()
    length = int(data[0])
    return json.loads(bytes(int(b) for b in data[1 : length + 1]).decode("utf-8"))


def _Ctx(seq_lengths: list[int], extra_kwargs: list[dict]) -> PoolerContext:
    """Build a context for a batch running without adapters."""
    return PoolerContext(seq_lengths=seq_lengths, extra_kwargs=extra_kwargs)


def test_loading_the_pooler_leaves_no_stub_behind(pooler_mod: ModuleType):
    """A leaked stub shadows the real adapter for every later test."""
    adapter = sys.modules.get("vllm_factory.pooling.vllm_adapter")
    assert adapter is None or hasattr(adapter, "VllmPoolerAdapter")
    assert pooler_mod.GLiNER25BoundaryPooler is not None


def test_empty_payload_decodes_as_an_empty_result(pooler_mod: ModuleType):
    assert _decode(pooler_mod._pack_json({}, torch.device("cpu"))) == {}


def test_empty_payloads_are_one_per_sequence(pooler_mod: ModuleType):
    payloads = pooler_mod._empty_payloads(3, torch.device("cpu"))

    assert len(payloads) == 3
    assert all(_decode(payload) == {} for payload in payloads)


def test_split_failure_still_answers_every_sequence(pooler_mod: ModuleType, monkeypatch):
    def _boom(hidden_states, seq_lengths):
        raise RuntimeError("bad lengths")

    monkeypatch.setattr(pooler_mod, "split_hidden_states", _boom)
    pooler = object.__new__(pooler_mod.GLiNER25BoundaryPooler)
    ctx = _Ctx([5, 7, 9], [{"a": 1}, {"b": 2}, {"c": 3}])

    outputs = pooler_mod.GLiNER25BoundaryPooler.forward(pooler, torch.zeros(21, 4), ctx)

    assert len(outputs) == 3
    assert all(_decode(payload) == {} for payload in outputs)


def test_missing_extras_do_not_shift_the_other_results(
    pooler_mod: ModuleType, monkeypatch
):
    monkeypatch.setattr(
        pooler_mod,
        "split_hidden_states",
        lambda hidden_states, seq_lengths: [torch.zeros(n, 4) for n in seq_lengths],
    )
    monkeypatch.setattr(
        pooler_mod.GLiNER25BoundaryPooler,
        "_decode_group",
        lambda self, sequences, extras: [
            pooler_mod._pack_json(extra, sequences[0].device) for extra in extras
        ],
    )
    pooler = object.__new__(pooler_mod.GLiNER25BoundaryPooler)
    # vLLM scheduled three sequences but only two carry extras.
    ctx = _Ctx([2, 3, 4], [{"first": True}, {}])

    outputs = pooler_mod.GLiNER25BoundaryPooler.forward(pooler, torch.zeros(9, 4), ctx)

    assert [_decode(payload) for payload in outputs] == [{"first": True}, {}, {}]


def test_same_key_shares_one_call_and_a_different_runtime_does_not(
    pooler_mod: ModuleType, monkeypatch
):
    """Grouping is (runtime, threshold, flags, max_len), and order is preserved."""
    monkeypatch.setattr(
        pooler_mod,
        "split_hidden_states",
        lambda hidden_states, seq_lengths: [torch.zeros(n, 4) for n in seq_lengths],
    )
    calls: list[list[str]] = []

    def _decode(self, sequences, extras):
        calls.append([extra["runtime"] for extra in extras])
        return [pooler_mod._pack_json({"i": index}, sequences[0].device) for index, _ in enumerate(extras)]

    monkeypatch.setattr(pooler_mod.GLiNER25BoundaryPooler, "_decode_group", _decode)
    pooler = object.__new__(pooler_mod.GLiNER25BoundaryPooler)
    shared = {"_compact": True, "runtime": "classic", "threshold": 0.5}
    other = {**shared, "runtime": "joint_ie"}
    ctx = _Ctx([2, 2, 2], [shared, other, shared])

    outputs = pooler_mod.GLiNER25BoundaryPooler.forward(pooler, torch.zeros(6, 4), ctx)

    assert calls == [["classic", "classic"], ["joint_ie"]]
    assert [_decode_payload(payload) for payload in outputs] == [{"i": 0}, {"i": 0}, {"i": 1}]


def _decode_payload(payload: torch.Tensor) -> dict:
    return _decode(payload)


def test_group_key_includes_runtime_threshold_and_max_len(pooler_mod: ModuleType):
    base = {
        "runtime": "classic",
        "threshold": 0.5,
        "include_confidence": False,
        "include_spans": True,
        "max_len": 32,
    }
    assert pooler_mod._group_key(base) == ("classic", 0.5, False, True, 32)
    changed = {**base, "runtime": "constrained_classification", "max_len": None}
    # A missing cap and an explicit None are the same group.
    assert pooler_mod._group_key(changed) == pooler_mod._group_key(
        {key: value for key, value in changed.items() if key != "max_len"}
    )
    assert pooler_mod._group_key(base) != pooler_mod._group_key(changed)
