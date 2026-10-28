# Phase 3: Model Architectures

This document specifies the architectures and the results of the initial structural testing for Phase 3 of the Mamba-CfC study on Ninapro DB1.

## 1. Implementations & Fallbacks

*   **Mamba Component**: We use a pure-PyTorch implementation of the Mamba architecture. 
    *   **Source:** [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py)
    *   **License:** MIT License.
    *   **Status:** Adopted as a fallback (`src/models/mamba_fallback.py`) since the official `mamba_ssm` requires complex CUDA toolchains which can be brittle on some hardware (e.g., Windows/RTX 4050 without compiled extensions). This implementation supports `pscan` for fast parallelized training and functionally mirrors the official implementation.
*   **CfC Component**: 
    *   **Source:** `ncps` (Neural Circuit Policies) PyTorch package.
    *   **Mode:** Plain default mode (`units=64`, `mode='default'`), using a fully-connected hidden state without NCP sparsity wiring.
    *   **Time Delta (`timespans`)**: Inspection of the `ncps` source code reveals that the `timespans` argument provides the scalar time-delta `ts` passed into the CfC cell update equations. When `timespans=None`, the library defaults to a uniform `ts = 1.0` for all timesteps. **Recommendation:** We freeze `timespans=None` (yielding `ts = 1.0`). Since our overlapping windows are extracted at a uniform stride, the data naturally represents uniform discrete timesteps, making dynamic time-scaling unnecessary.

## 2. Architectures

All models share the same sequence-level preprocessing, input embeddings, pooling mechanism, and classification head.

1.  **Mamba-only (Control)**: 
    *   Architecture: `Embedding (d=64) -> MambaBlocks (x2) -> Pooling -> Linear Head (53 classes)`.
    *   Purpose: Serves as the primary baseline.
2.  **CfC-only (Comparator)**:
    *   Architecture: `Embedding (d=64) -> CfC (units=64) -> Pooling -> Linear Head (53 classes)`.
    *   Purpose: Serves as the purely continuous-time baseline.
3.  **Mamba-CfC (Hybrid)**:
    *   Architecture: `Embedding (d=64) -> MambaBlocks (x2) -> Projection (d=64) -> CfC (units=64) -> Pooling -> Linear Head`.
    *   Purpose: The core experimental model.
4.  **Mamba-3L (depth control)**:
    *   Architecture: `Embedding (d=64) -> MambaBlocks (x3) -> Pooling -> Linear Head`.
    *   Purpose: An expanded depth control model (3 layers instead of 2).
5.  **Mamba-width-matched (d_model 88)**:
    *   Architecture: `Embedding (d=88) -> MambaBlocks (x2, dt_rank=6) -> Pooling -> Linear Head`.
    *   Purpose: An expanded width control model to match the hybrid's capacity.

*Note: Pooling strategy defaults to `last` timestep (matching the last-sample label rule) but is confirmed via `mean` pooling in evaluation.*

## 3. Structural Validation

A complete forward/backward structural test was run on a real batch of preprocessed data (Batch size: 32, Length: 30, Channels: 10). 

### Overfit Sanity Check & Latency

| Model | Parameters |
| :--- | :--- |
| **Mamba-only** (2 layers) | 69,557 |
| **CfC-only** | 53,685 |
| **Mamba-CfC** (Hybrid) | 123,253 |
| **Mamba-3L (depth control)** | 102,261 |
| **Mamba-width-matched (d_model 88)** | 122,373 |

**Notes on results:**
*   All models successfully map the inputs to the correct classification dimension without shape mismatches.
*   All models can learn (loss decreases over 50 steps), verifying that gradients successfully backpropagate through the pure-PyTorch Mamba block, the CfC cell, and the hybrid boundary.
*   Mamba-CfC parameter count (123k) is compared against the 3-layer Mamba model (102k) and width-matched model (122k), ensuring we have fair depth and width parameter controls, even though neither serves as a strict upper bound.
*   *GPU Memory tracking was omitted from the automated report because the test ran on the CPU fallback in this session, but memory is expected to be well within the limits of the RTX 4050 6GB VRAM for all architectures.*
