"""GLiNER2 vLLM Model — Schema-based multi-task extraction.

Uses custom vLLM-optimized DebertaV2EncoderModel as backbone (Flash DeBERTa Triton kernel
+ vLLM parallel layers) with GLiNER2Pooler for the head (SpanRep + CountLSTM + classifier + count_pred).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import ClassVar, Iterable, Tuple

import torch
import torch.nn as nn
from transformers import DebertaV2Config
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsLoRA

from vllm_factory.lora.routing import (
    BatchRouting,
    encoder_scope,
    find_punica_wrapper,
    sequence_slots,
    set_batch_routing,
)
from vllm_factory.packing import pad_batch, sequence_lengths
from vllm_factory.pooling.lora_heads import (
    SPAN_HEAD_NAMES,
    convert_heads_to_replicated,
    head_weights_mapper,
)
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

from .config import GLiNER2Config

logger = logging.getLogger(__name__)

# Load the custom DeBERTa v2 encoder with Flash DeBERTa Triton kernel
_ENCODER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "deberta_v2" / "deberta_v2_encoder.py"
)


def _import_deberta_v2_encoder():
    spec = importlib.util.spec_from_file_location("deberta_v2_encoder", str(_ENCODER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("deberta_v2_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_encoder_mod = _import_deberta_v2_encoder()
DebertaV2EncoderModel = _encoder_mod.DebertaV2EncoderModel
_ENCODER_PACKED_MODULES_MAPPING: dict[str, list[str]] = _encoder_mod.PACKED_MODULES_MAPPING
_ENCODER_EMBEDDING_MODULES: dict[str, str] = _encoder_mod.EMBEDDING_MODULES


class GLiNER2VLLMModel(nn.Module, SupportsLoRA):
    """GLiNER2 model for vLLM: custom vLLM-optimized encoder backbone + GLiNER2 pooler head.

    Declares `SupportsLoRA` and forwards the DeBERTa v2/v3 backbone's LoRA
    metadata under the ``encoder.`` prefix — the plugin wraps the encoder as
    ``self.encoder``, so every adapter target resolved by vLLM's LoRA manager
    walks through that attribute. PEFT adapters produced against the GLiNER2
    DeBERTa backbone (``target_modules=["query_proj", "key_proj",
    "value_proj"]`` by convention) are registered directly by layer name; no
    packing rewrite is needed. Task heads (``span_rep`` / ``classifier`` /
    ``count_pred`` / ``count_embed``) are converted to ``ReplicatedLinear`` so
    a multi-task-head adapter is loadable, and ``hf_to_vllm_mapper`` rewrites
    PEFT's top-level head prefixes onto ``_business_pooler.``.
    """

    is_pooling_model = True
    supports_lora: ClassVar[bool] = True
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = {
        f"encoder.{k}": [f"encoder.{n}" for n in v]
        for k, v in _ENCODER_PACKED_MODULES_MAPPING.items()
    }
    embedding_modules: ClassVar[dict[str, str]] = {
        f"encoder.{k}": v for k, v in _ENCODER_EMBEDDING_MODULES.items()
    }

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg: GLiNER2Config = vllm_config.model_config.hf_config
        self.config = cfg
        self.vllm_config = vllm_config

        # Build DeBERTa v2/v3 config for the custom encoder
        encoder_cfg = DebertaV2Config(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.encoder_hidden_size,
            num_hidden_layers=cfg.encoder_num_hidden_layers,
            num_attention_heads=cfg.encoder_num_attention_heads,
            intermediate_size=cfg.encoder_intermediate_size,
            hidden_act=cfg.encoder_hidden_act,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=cfg.encoder_max_position_embeddings,
            type_vocab_size=cfg.encoder_type_vocab_size,
            layer_norm_eps=cfg.encoder_layer_norm_eps,
            relative_attention=cfg.encoder_relative_attention,
            max_relative_positions=cfg.encoder_max_relative_positions,
            position_buckets=cfg.encoder_position_buckets,
            pos_att_type=cfg.encoder_pos_att_type,
            share_att_key=cfg.encoder_share_att_key,
            norm_rel_ebd=cfg.encoder_norm_rel_ebd,
            position_biased_input=cfg.encoder_position_biased_input,
            pad_token_id=cfg.encoder_pad_token_id,
        )

        # 1. Backbone — custom DeBERTa v2 with Flash DeBERTa Triton kernel
        self.encoder = DebertaV2EncoderModel(config=encoder_cfg)
        self._encoder_pad_id = int(cfg.encoder_pad_token_id or 0)

        # 2. GLiNER2 head (span_rep + count_embed + classifier + count_pred)
        # Imported here so register() stays import-safe without the gliner2 extra.
        from poolers.gliner2 import GLiNER2Pooler

        self._business_pooler = GLiNER2Pooler(
            hidden_size=cfg.encoder_hidden_size,
            max_width=cfg.max_width,
            counting_layer=cfg.counting_layer,
        )
        convert_heads_to_replicated(self._business_pooler, SPAN_HEAD_NAMES)
        self.hf_to_vllm_mapper = head_weights_mapper(SPAN_HEAD_NAMES)
        self.pooler = VllmPoolerAdapter(self._business_pooler, requires_token_ids=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM's pooling runner for embedding lookup."""
        return self.encoder.embeddings.word_embeddings(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Override sampling for pooling models — return empty outputs."""
        try:
            from vllm.sequence import SamplerOutput

            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the encoder over the scheduled batch and return flat hidden states.

        vLLM hands a single flat token tensor holding every scheduled sequence
        end to end. Feeding that straight to a bidirectional encoder would let
        the sequences attend to each other, so it is reshaped into one padded
        row per sequence before the forward and flattened back afterwards.

        Args:
            input_ids: Flat token ids for the whole batch, shape (total_tokens,).
            positions: Per-sequence position ids, restarting at 0 each sequence.
            intermediate_tensors: Unused; present for the vLLM model contract.
            inputs_embeds: Unused; embeddings are taken from ``input_ids``.
            **kwargs: Unused extras passed by the runner.

        Returns:
            Hidden states of shape (total_tokens, hidden_size), in the same
            token order as ``input_ids``.
        """
        flat = input_ids.view(-1) if input_ids.dim() > 1 else input_ids
        lengths = sequence_lengths(flat, positions)
        wrapper = find_punica_wrapper(self)
        slots = sequence_slots(wrapper, lengths)
        set_batch_routing(BatchRouting(wrapper=wrapper, slots=tuple(slots)))

        with torch.no_grad():
            if len(lengths) == 1:
                width = lengths[0]
                with encoder_scope(wrapper, slots, width):
                    hs = self.encoder(input_ids=flat[:width].unsqueeze(0))
            else:
                ids, mask = pad_batch(flat, lengths, self._encoder_pad_id)
                with encoder_scope(wrapper, slots, int(ids.shape[1])):
                    hs = self.encoder(input_ids=ids, attention_mask=mask)

        packed = torch.cat([hs[row, :length] for row, length in enumerate(lengths)], dim=0)
        # The runner pads the token count; give back a row per slot it sent.
        total = int(flat.numel())
        if packed.shape[0] == total:
            return packed
        out = packed.new_zeros((total, packed.shape[1]))
        out[: packed.shape[0]] = packed
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from GLiNER2 checkpoint.

        GLiNER2 state_dict prefixes:
            encoder.*        → self.encoder (custom DebertaV2EncoderModel)
            span_rep.*       → self.pooler.span_rep
            classifier.*     → self.pooler.classifier
            count_pred.*     → self.pooler.count_pred
            count_embed.*    → self.pooler.count_embed

        The custom encoder's load_weights() expects HF DeBERTa keys with
        'deberta.' prefix and handles the .attention.self. → .attention.self_attn.
        mapping internally.
        """
        encoder_prefix = "encoder."

        pooler_keys = set(self._business_pooler.state_dict().keys())

        backbone_weights = []
        pooler_loaded = {}

        for hf_name, tensor in weights:
            if hf_name.startswith(encoder_prefix):
                # Strip "encoder." and re-add "deberta." prefix for custom encoder
                hf_key = hf_name[len(encoder_prefix) :]

                # Handle vocab size mismatch (might have extra special tokens)
                if "word_embeddings.weight" in hf_key:
                    vocab_size = getattr(self.config, "vocab_size", None)
                    if vocab_size and tensor.shape[0] != vocab_size:
                        if tensor.shape[0] > vocab_size:
                            tensor = tensor[:vocab_size]
                        else:
                            extra = vocab_size - tensor.shape[0]
                            tensor = torch.cat(
                                [
                                    tensor,
                                    torch.randn(extra, tensor.shape[1]) * 0.02,
                                ],
                                dim=0,
                            )

                backbone_weights.append(("deberta." + hf_key, tensor))
            else:
                # Pooler weights: span_rep.*, classifier.*, count_pred.*, count_embed.*
                if hf_name in pooler_keys:
                    pooler_loaded[hf_name] = tensor

        # Load backbone via custom encoder's load_weights
        self.encoder.load_weights(backbone_weights)
        logger.info("[GLiNER2] Loaded encoder: %s weight tensors", len(backbone_weights))

        # Load pooler (use business pooler directly to avoid _inner. prefix mismatch).
        #
        # We keep ``strict=False`` because backbone / ``deberta.*`` keys are
        # loaded separately above — they'd otherwise show up as "unexpected"
        # here. But we still guard against the failure mode that masked the
        # count_lstm_v2 / count_lstm_moe incompatibility for months: with
        # the wrong ``GLiNER2Pooler.count_embed`` variant, every
        # ``count_embed.*`` key beyond the shared pos_embedding + GRU is
        # silently dropped and inference proceeds with a mostly random-init
        # count_embed. Compute expected / actual key sets from the pooler
        # + filtered weights and raise if anything was dropped.
        if pooler_loaded:
            missing = pooler_keys - pooler_loaded.keys()
            unexpected = pooler_loaded.keys() - pooler_keys
            if missing or unexpected:
                raise RuntimeError(
                    "GLiNER2 pooler weight-load mismatch — "
                    f"counting_layer={getattr(self.config, 'counting_layer', '?')!r} "
                    f"missing={sorted(missing)!r} unexpected={sorted(unexpected)!r}. "
                    "This indicates a pooler variant / checkpoint mismatch: "
                    "the pooler's count_embed class does not match the "
                    "checkpoint's counting_layer."
                )
            self._business_pooler.load_state_dict(pooler_loaded, strict=False)
            device = next(self.encoder.parameters()).device
            dtype = self.vllm_config.model_config.dtype
            self._business_pooler.to(device=device, dtype=dtype)
            logger.info(
                "[GLiNER2] Loaded pooler: %s/%s keys (counting_layer=%s)",
                len(pooler_loaded),
                len(pooler_keys),
                getattr(self.config, "counting_layer", "?"),
            )
        else:
            raise RuntimeError(
                "GLiNER2 pooler weights were empty after load_weights. "
                "A boundary checkpoint prepared as a span model produces a "
                "random-init head; refusing to serve it."
            )

        return set(name for name, _ in self.named_parameters())
