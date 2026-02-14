# Megatron-Bridge DSA Enablement Plan

## Goals
- Surface DeepSeek Sparse Attention configuration through Megatron-Bridge so downstream Hugging Face models can opt into DSA-aware training runs without manual patching.
- Provide conversion utilities that translate Hugging Face checkpoint metadata into Megatron DSA settings (e.g., top-k, indexer ranks).
- Guarantee backwards compatibility for existing dense-only pipelines.

## Implementation Outline
- **Configuration Plumbing**  
  - Extend bridge configuration schemas with a `dsa` stanza that mirrors Megatron’s `MLATransformerConfig` additions.  
  - Add CLI flags/env overrides so pipelines can toggle `warmup` or `sparse` modes at launch.
- **Checkpoint & State Mapping**  
  - Introduce helpers that initialize the DSA indexer weights when importing dense checkpoints, including an optional dense warmup phase to populate the lightning indexer.  
  - Update save/load paths to persist DSA-specific buffers (indexer KL stats, top-k history) alongside existing MLA caches.
- **Workflow Integration**  
  - Provide recipe templates demonstrating dense warmup scheduling followed by sparse fine-tuning within the bridge orchestration layer.  
  - Add validation hooks that report DSA sparsity statistics back to orchestrators for monitoring.

## Testing & Rollout
- Unit-test the new configuration serialization/deserialization paths with both dense-only and DSA-enabled configs.
- Add smoke tests that round-trip a Hugging Face model through Megatron-Bridge with `dsa_training_mode=sparse` and confirm reproducible outputs.
- Coordinate documentation updates so downstream teams understand the staged training process (dense warmup → sparse adaptation).
