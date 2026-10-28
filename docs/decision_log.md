# Decision Log

| Date | Decision | Options Considered | Decision Maker |
|---|---|---|---|
| 2026-10-06 | Mamba Implementation: Option 1 (Pure-PyTorch Fallback) | 1. Pure-PyTorch Fallback <br> 2. WSL2 with official package <br> 3. Windows pre-compiled wheels | User |
| 2026-10-06 | Project Directory Structure Setup | Proposed standard DS structure | User |
| 2026-10-06 | Labels: restimulus & rerepetition | stimulus vs restimulus | User |
| 2026-10-06 | Exercises: All (E1, E2, E3) | Select subset vs All | User |
| 2026-10-06 | Class Mapping | Rest=0, E1=1-12, E2=13-29, E3=30-52 (Total: 53) | User |
| 2026-10-06 | Filtering: None | Option A (None) vs Option B (Smoothing) | User |
| 2026-10-06 | Rest Handling: Keep all | Keep vs Downsample vs Exclude (Rebalance at train-time ~58% Rest) | User |
| 2026-10-06 | Phase 3 CfC Mode | Default CfC (64 units), NO NCP wiring | Default CfC vs NCP wiring | User |
| 2026-10-06 | Phase 3 Control | Mamba-only (option for param-matched wider Mamba) | Mamba-only vs Mamba+GRU | User |
| 2026-10-06 | Phase 3 Pooling | Config option (mean vs last timestep) | Mean vs Last | User |

### Phase 3/4 Review & Fixes
- **Test Indexing Fix**: In 	ests/test_data.py, start_idx was originally read as a float. This caused a TypeError: slice indices must be integers during the 	est_data_integrity test. Fixed by explicitly casting start_idx, label, subject, and exercise to int within the test.
- **Normalization Leakage Isolation**: The single normalization test was split into 	est_train_normalization_sanity (verifying ~0 mean and ~1 std with a 1e-2 tolerance) and a new strict 	est_no_normalization_leakage that actively scales val/test data and asserts the saved scalers remain identical.
- **Model Parameters**: The parameter counts are concentrated in the Mamba Backbone and CfC modules.
  - The previous control name 'Mamba-larger' was renamed to **Mamba-3L (depth control)** (102,261 params).
  - The previous 'Mamba-param-matched' was renamed to **Mamba-width-matched (d_model 88)** (122,373 params).
  - *Note:* Neither of these are true upper bounds for the hybrid (123,253 params), but they act as fair depth and width controls.
- **Pooling Clarification & Bug Fix**: In early benchmark tests, an incorrect script failed to propagate the pool_mode parameter during testing, causing a hallucinated/incorrect initial loss of 0.4. Re-benchmarking (which correctly passes the pool_mode argument) shows that pool_mode='last' returns an initial step 0 loss of ~3.96, perfectly matching the theoretical random-initialization loss of ln(53) = 3.97. 
  - **Decision**: last timestep pooling will be used as the default configuration because it matches the *last-sample label rule* (window label determined by the last sample).
  - **Phase 4 Update**: This was definitively confirmed via a controlled 2-epoch benchmark during early Phase 4 training. Last-timestep pooling (Val F1: 0.4286) strictly outperformed Mean pooling (Val F1: 0.3902).
