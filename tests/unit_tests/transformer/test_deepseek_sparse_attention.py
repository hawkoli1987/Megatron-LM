import torch

from megatron.core.transformer.deepseek_sparse_attention import DeepSeekSparseIndexer
from megatron.core.transformer.transformer_config import MLATransformerConfig


def _make_config(mode: str) -> MLATransformerConfig:
    return MLATransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        dsa_training_mode=mode,
        dsa_index_n_heads=4,
        dsa_index_head_dim=32,
        dsa_index_topk=3,
        dsa_kl_loss_weight=0.5,
    )


def _dummy_inputs(batch: int, seq: int, config: MLATransformerConfig):
    hidden = torch.randn(batch, seq, config.hidden_size)
    qr = torch.randn(batch, seq, config.q_lora_rank or config.hidden_size)
    query = torch.randn(seq, batch, config.num_attention_heads, config.qk_head_dim)
    key = torch.randn_like(query)
    causal = torch.full((batch, seq, seq), float("-inf"))
    causal = torch.triu(causal, diagonal=1)
    return hidden, qr, query, key, causal


def test_dsa_warmup_outputs_mask_is_none():
    torch.manual_seed(1)
    config = _make_config("warmup")
    indexer = DeepSeekSparseIndexer(config, layer_number=1)
    hidden, qr, query, key, causal = _dummy_inputs(batch=2, seq=5, config=config)

    result = indexer(
        hidden_states=hidden,
        qr_states=qr,
        freqs_cis=None,
        causal_mask=causal,
        query=query,
        key=key,
        attn_scale=1.0,
        training=True,
    )

    assert result is not None
    assert result.additive_mask is None
    assert result.topk_indices.shape == (2, 5, config.dsa_index_topk)
    assert torch.isfinite(result.kl_loss)


def test_dsa_sparse_outputs_mask_applies_topk():
    torch.manual_seed(2)
    config = _make_config("sparse")
    indexer = DeepSeekSparseIndexer(config, layer_number=1)
    hidden, qr, query, key, causal = _dummy_inputs(batch=1, seq=4, config=config)

    result = indexer(
        hidden_states=hidden,
        qr_states=qr,
        freqs_cis=None,
        causal_mask=causal,
        query=query,
        key=key,
        attn_scale=1.0,
        training=True,
    )

    assert result is not None
    assert result.additive_mask is not None
    mask = result.additive_mask
    assert mask.shape == (1, 4, 4)

    topk = result.topk_indices[0]
    for row in range(topk.size(0)):
        selected = topk[row]
        assert torch.all(mask[0, row, selected] == 0)
        dropped_selector = torch.ones(mask.size(-1), dtype=bool)
        dropped_selector[selected] = False
        dropped_values = mask[0, row, dropped_selector]
        if dropped_values.numel() > 0:
            assert torch.all(dropped_values <= 0)
