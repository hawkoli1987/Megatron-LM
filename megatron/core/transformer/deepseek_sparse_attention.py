import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig


_EPS = 1e-6


def _hadamard_transform(x: Tensor) -> Tensor:
    """Apply an in-place Hadamard transform along the last dimension."""
    hidden = x.size(-1)
    if hidden & (hidden - 1) != 0:
        raise ValueError(
            f"DSA requires index head dimension to be a power of two, but got {hidden}."
        )
    stride = 1
    out = x
    while stride < hidden:
        out = out.reshape(*out.shape[:-1], -1, 2, stride)
        first = out[..., 0, :]
        second = out[..., 1, :]
        out = torch.cat((first + second, first - second), dim=-1)
        stride <<= 1
    return out / math.sqrt(hidden)


def _apply_rotary(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary positional embedding to the provided tensor."""
    dtype = x.dtype
    complex_x = torch.view_as_complex(
        x.float().view(*x.shape[:-1], -1, 2)
    )
    freqs = freqs_cis.view(1, x.size(1), 1, -1)
    rotated = torch.view_as_real(complex_x * freqs).flatten(-2)
    return rotated.to(dtype)


@dataclass
class DSAOutput:
    topk_indices: Tensor
    additive_mask: Optional[Tensor]
    kl_loss: Tensor


class DeepSeekSparseIndexer(nn.Module):
    """Implements the DeepSeek Sparse Attention indexer for training."""

    def __init__(self, config: MLATransformerConfig, layer_number: int) -> None:
        super().__init__()
        if config.tensor_model_parallel_size != 1:
            raise ValueError("DSA training currently requires tensor_model_parallel_size == 1.")

        self.config = config
        self.layer_number = layer_number

        self.training_mode = getattr(config, "dsa_training_mode", "disabled")
        self.index_heads = getattr(config, "dsa_index_n_heads", 64)
        self.head_dim = getattr(config, "dsa_index_head_dim", 128)
        self.topk = getattr(config, "dsa_index_topk", 2048)
        self.loss_weight = getattr(config, "dsa_kl_loss_weight", 1.0)
        self.detach_inputs = getattr(config, "dsa_detach_indexer_inputs", True)
        self.log_metrics = getattr(config, "dsa_log_aux_metrics", False)

        self.rope_dim = min(config.qk_pos_emb_head_dim, self.head_dim)

        self.q_proj = nn.Linear(config.q_lora_rank, self.index_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layernorm_epsilon)
        self.weight_proj = nn.Linear(config.hidden_size, self.index_heads, bias=False)

    def _maybe_detach(self, tensor: Tensor) -> Tensor:
        return tensor.detach() if self.detach_inputs else tensor

    def _compute_index_scores(
        self,
        hidden_states: Tensor,
        qr_states: Tensor,
        freqs_cis: Optional[Tensor],
    ) -> Tensor:
        bsz, seqlen, _ = hidden_states.shape

        q_latent = self.q_proj(qr_states).view(bsz, seqlen, self.index_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q_latent, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        if self.rope_dim > 0 and freqs_cis is not None:
            q_pe = _apply_rotary(q_pe, freqs_cis)
        q_latent = torch.cat((q_pe, q_nope), dim=-1)

        k_latent = self.k_proj(hidden_states)
        k_latent = self.k_norm(k_latent)
        if self.rope_dim > 0 and freqs_cis is not None:
            k_pe, k_nope = torch.split(
                k_latent, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            k_pe = _apply_rotary(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
            k_latent = torch.cat((k_pe, k_nope), dim=-1)

        q_latent = _hadamard_transform(q_latent)
        k_latent = _hadamard_transform(k_latent)

        weights = self.weight_proj(hidden_states) * (self.index_heads ** -0.5)

        raw = torch.einsum("bqhd,bkd->bqhk", q_latent, k_latent)
        raw = F.relu(raw)
        index_scores = (raw * weights.unsqueeze(-1)).sum(dim=2)
        return index_scores

    def forward(
        self,
        hidden_states: Tensor,
        qr_states: Tensor,
        freqs_cis: Optional[Tensor],
        causal_mask: Optional[Tensor],
        *,
        attention_scores: Optional[Tensor] = None,
        query: Optional[Tensor] = None,
        key: Optional[Tensor] = None,
        attn_scale: Optional[float] = None,
        training: bool,
    ) -> Optional[DSAOutput]:
        if self.training_mode == "disabled":
            return None

        hidden_states = self._maybe_detach(hidden_states)
        qr_states = self._maybe_detach(qr_states)

        bsz, seqlen, _ = hidden_states.shape
        index_scores = self._compute_index_scores(hidden_states, qr_states, freqs_cis)

        if causal_mask is not None:
            index_scores = index_scores + causal_mask.unsqueeze(0)

        if query is not None and key is not None:
            q = query.permute(1, 0, 2, 3).float()
            k = key.permute(1, 0, 2, 3).float()
            head_dim = min(q.size(-1), k.size(-1), self.config.qk_head_dim)
            q = q[..., :head_dim]
            k = k[..., :head_dim]
            dense_logits = torch.einsum("bshd,bthd->bsht", q, k)
            if attn_scale is not None:
                dense_logits = dense_logits * attn_scale
            if causal_mask is not None:
                dense_logits = dense_logits + causal_mask.unsqueeze(0).unsqueeze(2)
            dense_probs = torch.softmax(dense_logits, dim=-1).detach()
            dense_probs = dense_probs.mean(dim=2)
        elif attention_scores is not None:
            dense_logits = attention_scores.float()
            if causal_mask is not None:
                dense_logits = dense_logits + causal_mask.unsqueeze(0).unsqueeze(2)
            dense_probs = torch.softmax(dense_logits, dim=-1).detach()
            dense_probs = dense_probs.mean(dim=2)
        else:
            dense_probs = torch.softmax(index_scores, dim=-1).detach()

        topk = min(self.topk, seqlen)
        topk_indices = torch.topk(index_scores, k=topk, dim=-1).indices

        if self.training_mode == "warmup":
            pred_probs = torch.softmax(index_scores, dim=-1)
            target_probs = dense_probs
            kl = F.kl_div(
                torch.log(pred_probs + _EPS), target_probs + _EPS, reduction="batchmean"
            )
            additive_mask = None
        else:
            gathered_pred = torch.gather(index_scores, dim=-1, index=topk_indices)
            pred_probs = torch.softmax(gathered_pred, dim=-1)
            target_selected = torch.gather(dense_probs, dim=-1, index=topk_indices)
            target_selected = target_selected / (target_selected.sum(dim=-1, keepdim=True) + _EPS)
            kl = F.kl_div(
                torch.log(pred_probs + _EPS), target_selected + _EPS, reduction="batchmean"
            )
            additive_mask = index_scores.new_full((bsz, seqlen, seqlen), float("-inf"))
            additive_mask.scatter_(-1, topk_indices, 0.0)

        if training and self.loss_weight > 0.0:
            scaled_loss = self.loss_weight * kl
            index_scores = MoEAuxLossAutoScaler.apply(index_scores, scaled_loss)

        if training and self.log_metrics:
            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            save_to_aux_losses_tracker(
                "dsa_indexer_loss",
                kl.detach(),
                self.layer_number,
                num_layers,
                reduce_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )

        return DSAOutput(
            topk_indices=topk_indices,
            additive_mask=additive_mask,
            kl_loss=kl.detach(),
        )
