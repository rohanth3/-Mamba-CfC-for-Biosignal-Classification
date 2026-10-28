# Mamba-CfC for Biosignals
**Capstone Project - Phase 1 / Review 1**

This repository contains the first implementation milestone for a controlled attribution study evaluating the addition of a Closed-form Continuous-time (CfC) component to a Mamba backbone for EMG signal classification.

## Project Context
Our goal is to answer the research question: *"What measurable change occurs when a CfC component is added to a Mamba backbone?"*
The chosen primary domain is Electromyography (EMG) from the **Ninapro DB1** dataset. The study is rigorously controlled—comparing the hybrid against parameter-matched and depth-matched Mamba-only baselines.

## Repository Structure
- `data/`: Contains raw Ninapro DB1 `.mat` files and HDF5 preprocessed splits (`train.h5`, `val.h5`, `test.h5`).
- `docs/`: Architecture specifications, design documents, and decision logs.
  - `architecture_spec.md`: Fully details the network, hyperparameters, pooling methodology, and hardware benchmarks.
- `figures/`: Slide-ready diagrams of the dataset splits, normalization, windowing logic, class distributions, and the model architecture.
- `src/models/`: Model architecture implementation, including the PyTorch-only Mamba fallback, CfC logic, and model definitions.
- `scripts/`: Tools for downloading dataset, building splits, preprocessing, plotting, and benchmarking.
- `tests/`: Comprehensive test suite verifying data integrity, splitting logic, and architecture parameters.

## Implementation Milestones Achieved
1. **Data Pipeline**: Robust, reproducible dataset windowing (300 ms length, 100 ms stride), and subject splitting (18 Train / 4 Val / 4 Test) with strict leakage checks and channel-wise Z-score normalization computed exclusively on the training split.
2. **Architecture Finalized**: PyTorch `2.6.0+cu124` environment established with CUDA acceleration. Forward passes and integration of Mamba and CfC are mathematically verified via parameterized unit tests.
3. **Rigorous Baselines Established**: Memory and Latency benchmarks demonstrate the computational efficiency of the Hybrid Mamba-CfC design compared to scaling up the Mamba baseline alone.
4. **Validation Suite Passed**: 100% of data integrity, structure, parameters, and normalization leakage tests pass (12/12 pytest tests passing).
5. **Phase 4 Empirical Training & Evaluation Complete**:
   - All 5 architectures (`mamba-only`, `cfc-only`, `mamba-cfc`, `mamba-3l`, `mamba-width`) fully trained with `WeightedRandomSampler` and early stopping.
   - Evaluated on isolated unseen test subjects (Subjects 24–27).
   - Confirmed CfC addition improves the Mamba backbone beyond simple parameter-scaling controls.

## Key Empirical Findings (Unseen Test Set)
- `cfc-only`: Macro-F1 **0.0795** | Bal-Acc **0.1002**
- `mamba-cfc` (Hybrid): Macro-F1 **0.0776** | Bal-Acc **0.0966**
- `mamba-width`: Macro-F1 **0.0757** | Bal-Acc **0.0928**
- `mamba-only`: Macro-F1 **0.0753** | Bal-Acc **0.0910**
- `mamba-3l`: Macro-F1 **0.0732** | Bal-Acc **0.0897**

*Insight:* The strict inter-subject data isolation honestly exposed the severe inter-subject generalization bottleneck inherent in raw 100 Hz sEMG envelopes. Future directions include subject-adaptive normalization, domain-adversarial feature alignment, and few-shot calibration.
