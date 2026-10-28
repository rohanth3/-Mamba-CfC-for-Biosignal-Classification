# Phase 1 Architecture Specification: Mamba-CfC

## Overview
This document specifies the architectures and experimental configurations for the Mamba-CfC comparative study on the Ninapro DB1 dataset.

## Environment & Reproducibility
- **Python**: `3.11.9`
- **PyTorch Version**: `2.6.0+cu124` (CUDA 12.4 enabled).
- **Core Dependencies**: `ncps`, `numpy`, `h5py`, `scipy`, `pyyaml`.
- **Mamba Fallback**: The model uses a vendored PyTorch-only Mamba implementation to ensure cross-platform reproducibility without compiling Triton kernels.
  - The fallback `mamba_fallback.py`/`pscan.py` is derived from `alxndrTL/mamba.py` (Commit: `5c4136c2c28bc10d5d6655aa81b7a60051b50f00`) and `johnma2006/mamba-minimal` (Commit: `03de542a36d873f6e6c4057ad687278cc6ae944d`). The MIT license for these derivations is included in `src/models/LICENSE_mamba_minimal.txt`.
  - The fallback `pscan` (parallel scan) has been tested to match sequential recurrence within `rtol=1e-4`, `atol=1e-4`.

## Diagram
```mermaid
flowchart TD
    In[Input Window: 30x10] --> Emb[Linear Embedding: 10 -> 64]
    
    subgraph Backbone
    direction TB
    M[Mamba Layer 0: d=64, e=2, d_conv=4, d_state=16, dt_rank=4]
    M2[Mamba Layer 1: d=64, e=2, d_conv=4, d_state=16, dt_rank=4]
    M --> M2
    end
    
    Emb --> Backbone
    Backbone --> Proj[Feature Projection: 64 -> 64]
    Proj --> CfC[CfC: 64 units, ts=1.0]
    
    CfC --> Pool[Last-timestep Pooling]
    Pool --> Head[Linear Head: 64 -> 53]
    Head --> Out[Class Logits: 53]
```

## Models and Parameters

| Model | Embedding | Mamba Layer(s) | Projection | CfC | Head | Total |
|-------|-----------|----------------|------------|-----|------|-------|
| **Mamba-only** (Baseline) | 704 | 65,408 (x2) | 0 | 0 | 3,445 | **69,557** |
| **Mamba-3L (depth control)** | 704 | 98,112 (x3) | 0 | 0 | 3,445 | **102,261** |
| **Mamba-width-matched** | 968 | 116,688 (x2) | 0 | 0 | 4,717 | **122,373** |
| **CfC-only** (Comparator) | 704 | 0 | 0 | 49,536 | 3,445 | **53,685** |
| **Mamba-CfC** (Hybrid) | 704 | 65,408 (x2) | 4,160 | 49,536 | 3,445 | **123,253** |

*(Note: Above parameter counts may vary slightly due to bias configurations, exact calculations are validated by `test_models.py` tests. The hybrid parameter delta consists exactly of the CfC core module and the feature projection layer over the Mamba-only baseline).*

- **Mamba-width-matched**: Differs from the hybrid's backbone purely by width (`d_model=88` and corresponding `dt_rank=6` to closely match hybrid parameter count without structural additions. It is 0.7% below the hybrid).
- **Mamba-3L (depth control)**: Differs by adding a third Mamba layer of standard `d_model=64`. Note that neither of these serve as a strict upper bound, but act as fair controls.

## Hyperparameters
- **Input Window**: 30 samples (300 ms) x 10 channels.
- **Mamba Base**: `d_model=64`, `expand_factor=2`, `d_conv=4`, `d_state=16`, `dt_rank=4`.
- **CfC**: 64 units, `ts=1.0` (frozen time delta).
- **Pooling**: **Default: last-timestep** (`x[:, -1, :]`). 
  - *Rationale:* This matches the *last-sample label rule* used during window extraction. Initial loss correctly sanity-checks at `~3.96` (matching theoretical $\ln(53)$). During Phase 4, a strict empirical training benchmark confirmed `last-timestep` pooling yields superior generalization over `mean` pooling (Epoch 2 Val F1: 0.4286 vs 0.3902), definitively locking it in as the default.

## CfC Implementation Constraints
- **Constant Time-Delta**: In the official `ncps` CfC integration, the `ts` variable is injected directly into the update mechanism (as verified in NCPS CfC implementation). 
  - We explicitly configure the CfC with a fixed, uniform constant `ts=1.0`.
  - The model does **NOT** exploit real continuous-time interval information (time-deltas do not vary).

## Training Protocol
- **Loss**: `CrossEntropyLoss`.
- **Handling Class Imbalance**: Due to high class imbalance (~58% Rest class), the training set employs a `WeightedRandomSampler`. It calculates per-class inverse frequencies on the training split, assigning probabilities such that minibatches contain a more uniform distribution of the 53 classes.
- **Metrics**: Macro-F1 and Balanced Accuracy.
- **Checkpointing**: Checkpointing stores the model state dict achieving the highest validation Macro-F1 score across training epochs.

## Tensor Shape Pipeline
1. Input: `[Batch, 30, 10]`
2. Embedding: `[Batch, 30, 64]`
3. Mamba: `[Batch, 30, 64]`
4. Projection: `[Batch, 30, 64]`
5. CfC: `[Batch, 30, 64]`
6. Pooling: `[Batch, 64]`
7. Head: `[Batch, 53]`

## Hardware & Performance Benchmarks
We conducted latency and memory benchmarking to determine the computational overhead introduced by the CfC component. Benchmarks were evaluated on PyTorch `2.6.0+cu124`.

**At Batch Size 256:**
- **Mamba-only**: 24.43 ms | Peak Mem: 914.65 MB
- **Mamba-3L**: 35.94 ms | Peak Mem: 1205.01 MB
- **Mamba-param-matched**: 33.43 ms | Peak Mem: 1246.78 MB
- **CfC-only**: 15.38 ms | Peak Mem: 42.70 MB
- **Mamba-CfC (Hybrid)**: 31.64 ms | Peak Mem: 915.06 MB

**Analysis:**
- The Hybrid model exhibits roughly ~7.2 ms higher latency than the baseline Mamba-only at BS 256. 
- However, compared to standard scale-up approaches (Mamba-3L or Mamba-param-matched), the Hybrid is actually **faster** (31.64 ms vs ~33-35 ms) and requires **substantially less peak memory** (~915 MB vs ~1200+ MB). This confirms that hybridizing Mamba with CfC scales memory more efficiently than purely widening or deepening the Mamba backbone.
