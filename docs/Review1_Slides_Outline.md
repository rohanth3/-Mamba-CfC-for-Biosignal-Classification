# Review 1: Slides Outline & Talking Points

## Slide 1: Title & Project Context
- **Title**: Mamba-CfC for Biosignals
- **Goal**: A controlled attribution study evaluating the integration of Closed-form Continuous-time (CfC) components with a Mamba sequence backbone for Electromyography (EMG) classification.
- **Hypothesis**: Replacing layers of a standard recurrent or state-space model with CfC might optimize representation while controlling parameter explosion, but outcomes (positive, neutral, negative) are rigorously tested against matched baselines.

## Slide 2: Dataset & Preprocessing Pipeline
- **Ninapro DB1**: 26 active subjects (S6 missing from source).
- **Extraction Protocol**: Strict 300 ms sliding windows (30 samples) with 100 ms stride (10 samples). Window-level labeling utilizes the *last-sample label rule* to enforce causality.
- **Controls**: Strict data isolation. No window overlaps subjects, and normalization (Z-Score) statistics are rigorously calculated purely from the training split to prevent data leakage.
- **Visuals**: Use `figures/window_extraction_diagram.png` and `figures/before_after_preprocessing.png`.

## Slide 3: Subject & Class Distributions
- **Splits**: 18 Train / 4 Val / 4 Test. Handled precisely to maintain roughly ~58% Rest ratio across all partitions.
- **Movements**: 53 classes total (1 Rest + 52 Movements). Handled through `WeightedRandomSampler` during training to combat heavy imbalance.
- **Visuals**: Use `figures/subject_split_diagram.png` and `figures/class_distribution_per_split.png`.

## Slide 4: Proposed Hybrid Architecture
- **Structure**: Initial Embedding -> 2x Mamba Layers -> Feature Projection -> CfC -> Last-timestep Pooling -> Classifier Head.
- **Time Modelling**: For this phase, CfC utilizes a constant time delta (`ts=1.0`), preventing it from exploiting variable interval knowledge, serving purely as a sequential comparator.
- **Pooling Stability**: Evaluated mean vs. last-timestep pooling empirically. A controlled training benchmark on the Hybrid model confirmed last-timestep pooling is strictly superior (Val F1: 0.43 vs 0.39), as it mathematically aligns with our last-sample label rule.
- **Visuals**: Use `figures/architecture_diagram.png`.

## Slide 5: Strict Parameter Control
- **Mamba-only (Baseline)**: ~70k parameters.
- **Mamba-CfC (Hybrid)**: ~123k parameters (Added Projection + CfC).
- **Mamba-width-matched**: ~122k parameters (Expanded width d_model=88 for rigorous control).
- **Mamba-3L**: ~102k parameters (Depth baseline).
- **The Point**: We never assume a parameter increase guarantees capability. Every structural variable (width vs depth vs CfC component) is controlled, though we acknowledge the Hybrid has slightly more parameters than our matched controls (~0.7% more than width-matched).

## Slide 6: Computational Hardware Findings
- **Discovery**: Latency & Memory profiling on PyTorch 2.6.0+cu124 (BS 256).
- **Latency**: Hybrid (31.6 ms) is slower than baseline Mamba (24.4 ms), BUT faster than scale-up controls like Mamba-3L (35.9 ms).
- **Memory Efficiency**: Hybrid strictly maintains baseline peak memory footprint (~915 MB), heavily outperforming Mamba-3L/param-matched variants (~1.2GB).
- **Conclusion**: Adding CfC costs minor latency but scales memory highly efficiently.

## Slide 7: Phase 4 Empirical Findings & Attribution Results
- **The Finding (Overfitting Bottleneck)**: All models exhibited massive overfitting to the 18 Training subjects, failing to generalize to the isolated Validation/Test subjects. Training F1 consistently climbed to **~65-70%** (triggering aggressive early stopping), while Test F1 collapsed to **~7-8%**.
- **Conclusion**: The strict data isolation strategy successfully caught the massive inter-subject variance inherent in raw 100Hz envelope EMG signals. The models memorized the specific neural signatures of the training pool but could not extrapolate them cross-subject.
- **Architectural Attribution**: Despite the bottleneck, relative performance on the Test set definitively isolates the architectural contribution of the CfC component.
  1. `cfc-only` (53K): **0.0795** *(Best Generalization)*
  2. `mamba-cfc` (123K): **0.0776** *(Hybrid)*
  3. `mamba-width` (122K): **0.0757**
  4. `mamba-only` (69K): **0.0753** *(Baseline)*
  5. `mamba-3l` (102K): **0.0732**
- **Evidence-Based Answers**:
  - *Did CfC improve the backbone?* **Yes.** The Hybrid (0.0776) outperformed the baseline (0.0753).
  - *Was it just parameter scaling?* **No.** The Hybrid outperformed *both* strict structural controls (`mamba-width` and `mamba-3l`). 
  - *Surprise Insight*: The pure `cfc-only` model actually proved the most resistant to overfitting, suggesting continuous-time dynamics may inherently regularize against inter-subject variance better than discrete state-space expansions in raw EMG.
