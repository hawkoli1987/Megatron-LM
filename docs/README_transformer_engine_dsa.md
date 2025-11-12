# Transformer Engine DSA Integration Plan

## Objectives
- Add sparse token selection support to the MLA attention kernels so they can consume DSA top-k masks from Megatron-LM during both forward and backward passes.
- Provide a lightweight indexer training API that exposes lightning indexer activations and gradients without duplicating projections in Python.
- Maintain feature parity across Hopper and Blackwell kernels while keeping existing dense-path regression tests green.

## Proposed Workstreams
- **Kernel Extensions**  
  - Extend MLA fused attention kernels to accept an additional per-query sparse mask (block-structured) and fallback to dense behavior when the mask is absent.  
  - Add optional FP8 blockwise quantization utilities shared with the inference repo to avoid reimplementing quantization logic.
- **Indexer Autograd Support**  
  - Export custom autograd functions that surface the KL-divergence loss hooks currently implemented in Python so Megatron can attach DSA losses without manual bookkeeping.  
  - Ensure gradients can be accumulated independently for the indexer parameters while preventing cross-talk with the dense attention path.
- **Configuration & Testing**  
  - Introduce a `dsa_mode` flag mirroring the Megatron configuration to make staged roll-out (disabled → warmup → sparse) explicit in TE.  
  - Add unit tests for warmup and sparse modes including mask broadcasting edge cases and distributed TP=1/2 validation.

## Dependencies & Coordination
- Align Tensor shapes and scaling factors with Megatron’s `DeepSeekSparseIndexer` to avoid redundant transpose/cast operations.
- Coordinate release timelines with Megatron-LM so the kernel API lands before the training-side default flips to sparse mode.
- Work with QA to extend existing MLA CI coverage (nightly Hopper + Blackwell) to include sparse attention regression suites.
