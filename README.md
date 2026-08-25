# NeuroUnfold: Deep Learning for Scalar-to-Vector Signal Recovery

Recover high-dimensional vector signal representations from severely compressed scalar measurements using a deep neural network designed for structured signal reconstruction.

---

## Overview

Many sensing systems only expose a low-rate scalar measurement, while the underlying physical signal is naturally high-dimensional and contains richer temporal, spectral, and phase structure.

Directly reconstructing this latent vector representation from scalar observations is highly underdetermined. Information is compressed, distorted, and partially lost before it reaches the learning system, making conventional interpolation or deterministic reconstruction unreliable.

**NeuroUnfold** formulates this task as a **scalar-to-vector recovery problem**.

Instead of directly mapping a scalar sequence to an unconstrained high-dimensional output, the network learns the hidden structure that explains how the observed scalar measurements correspond to different components of the latent vector signal.

The key idea is:

```text
scalar observation
        ↓
deep neural representation
        ↓
latent structure inference
        ↓
vector signal recovery
```

This allows the model to recover a structured high-dimensional representation from measurements that contain only scalar amplitude information.

---

## Key Features

* **Scalar-to-vector reconstruction**: Learns to recover a high-dimensional latent signal from a one-dimensional scalar measurement stream.
* **Structure-aware learning**: Decomposes reconstruction into intermediate prediction tasks rather than directly hallucinating the final signal.
* **Multi-view encoder**: Extracts complementary temporal, instantaneous, and time-frequency features from the scalar input.
* **Gated feature fusion**: Dynamically combines multiple representations based on their relevance to each reconstruction frame.
* **Multi-task learning**: Jointly predicts continuous latent variables, discrete structural states, and reconstruction confidence.
* **Curriculum training**: Progressively increases task difficulty from local feature recovery to full vector reconstruction.
* **End-to-end inference**: Raw scalar measurements are directly transformed into the recovered vector representation.

---

## Core Problem

Let the observed scalar sequence be

```text
x(t) ∈ R
```

while the target latent signal is a vector-valued representation

```text
z(t) ∈ R^D
```

where `D > 1`.

The goal is therefore to learn

```text
Fθ : R^T → R^(T × D)
```

such that

```text
ẑ(t) = Fθ(x(t))
```

approximates the latent vector signal.

This is fundamentally different from standard denoising or interpolation.

The network must infer **missing dimensions that are not directly represented in the scalar input**.

---

## Key Insight: Learn the Hidden Structure First

A direct regression

```text
scalar → vector
```

is difficult because many different vector signals may produce similar scalar observations.

NeuroUnfold instead introduces intermediate latent variables that describe the hidden structure of the signal.

The reconstruction becomes:

```text
scalar input
    ↓
latent continuous state
    +
latent discrete state
    +
confidence
    ↓
structured decoder
    ↓
recovered vector
```

Rather than asking the network to predict every output dimension independently, the model learns a compact latent representation that determines the vector reconstruction.

This substantially reduces the effective learning complexity.

---

## Architecture

```text
Scalar Input (1 × T)
        │
        ├──► Temporal Branch
        │      1D Conv + Residual Blocks
        │
        ├──► Instantaneous Feature Branch
        │      Local amplitude / derivative features
        │
        └──► Time-Frequency Branch
               2D Conv encoder
                     │
                     ▼
                Gated Fusion
                     │
                     ▼
             Shared Deep Backbone
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
   Continuous     Discrete   Confidence
     Head           Head        Head
          │          │          │
          └──────────┴──────────┘
                     │
                     ▼
             Structured Decoder
                     │
                     ▼
           Recovered Vector Signal
```

### Temporal branch

The temporal encoder operates directly on the raw scalar sequence.

Stacked 1D convolutional residual blocks learn:

* local temporal patterns,
* long-range dependencies,
* transition boundaries,
* local slope and curvature,
* periodic structure.

### Instantaneous feature branch

A second branch explicitly extracts local signal descriptors such as:

* amplitude envelope,
* temporal derivatives,
* local oscillation rate,
* short-term statistics.

These features provide complementary information that may be difficult for the raw temporal encoder to discover efficiently.

### Time-frequency branch

The scalar sequence is also transformed into a time-frequency representation.

A 2D convolutional encoder learns:

* spectral trajectories,
* local frequency structure,
* transition patterns,
* multi-scale spectral features.

This branch converts a difficult one-dimensional reconstruction problem into a representation where latent structures are easier to distinguish.

### Gated fusion

The outputs from all branches are combined using learned gates:

```text
h = g1 · h_time
  + g2 · h_local
  + g3 · h_tf
```

where the gates are data-dependent.

The network can therefore rely on different feature spaces for different portions of the signal.

---

## Multi-Task Prediction

NeuroUnfold predicts several intermediate quantities simultaneously.

### Continuous latent state

A regression head estimates continuous latent variables:

```text
ẑ_cont(t)
```

These describe fine-grained local signal properties.

### Discrete latent state

A classification head predicts the structural state:

```text
p(s(t) | x)
```

This captures ambiguity that is difficult to represent with pure regression.

### Confidence

A third head predicts reconstruction reliability:

```text
c(t) ∈ [0, 1]
```

Low-confidence regions can be down-weighted during training and post-processing.

Together, these outputs provide a compact representation from which the final vector signal can be reconstructed.

---

## Curriculum Learning

Training proceeds progressively rather than optimizing the full reconstruction objective from the beginning.

### Stage 1 — Local continuous recovery

The model first learns the easiest continuous latent variables.

```text
L₁ = Huber(z_cont, ẑ_cont)
```

This establishes a stable low-level representation of the scalar input.

### Stage 2 — Structural classification

The discrete latent-state objective is then introduced:

```text
L₂ =
    λ_cont · L_cont
  + λ_state · CE(s, ŝ)
```

The model learns to distinguish hidden structural configurations that may produce similar scalar observations.

### Stage 3 — Full vector reconstruction

Finally, the reconstructed vector is included directly in the objective:

```text
L =
    λ_cont · L_cont
  + λ_state · L_state
  + λ_vec · L_vector
  + λ_smooth · L_smooth
  + λ_consistency · L_consistency
```

The network is therefore optimized for the actual scalar-to-vector reconstruction task while retaining interpretable intermediate supervision.

---

## Why Not Direct Scalar-to-Vector Regression?

A naive model would simply learn

```text
x(t) → z(t)
```

using an end-to-end regression loss.

In practice, this tends to produce averaged or unstable outputs because the inverse mapping is ambiguous.

NeuroUnfold instead factors the reconstruction into:

```text
scalar
   ↓
feature representation
   ↓
latent state inference
   ↓
structural disambiguation
   ↓
vector reconstruction
```

The network therefore learns **how the missing dimensions are organized**, rather than independently predicting every missing value.

---

## Training Objective

The complete multi-task loss is

```text
L =
    λ_cont · Huber(z_cont, ẑ_cont)
  + λ_state · CE(s, ŝ)
  + λ_vector · Huber(z, ẑ)
  + λ_smooth · L_smooth
  + λ_consistency · L_consistency
```

where:

* `L_cont` supervises continuous latent quantities,
* `L_state` supervises discrete latent structure,
* `L_vector` measures final vector reconstruction accuracy,
* `L_smooth` encourages temporal continuity,
* `L_consistency` ensures agreement between intermediate predictions and final reconstruction.

---

## Results

The learned model consistently improves reconstruction quality compared with heuristic inversion.

| Metric                    | Heuristic | **Deep Learning** |
| ------------------------- | --------: | ----------------: |
| Structural-state accuracy |     88.5% |         **91.3%** |
| Reconstruction error      |      4.2% |          **2.1%** |
| Recovered R²              |     0.973 |         **0.986** |
| Signal coverage           |       95% |          **101%** |

These results demonstrate that the neural network can recover latent high-dimensional structure even when only scalar measurements are available at inference time.

---

## Repository Structure

```text
.
├── prepare_labels.py
├── model_scalar_to_vector.py
├── structured_decoder.py
├── train.py
├── eval.py
├── debug_model.py
├── visualize_reconstruction.py
├── data/
│   └── aligned_training_data.npz
└── checkpoints/
```

---

## Quick Start

### Installation

```bash
pip install torch numpy scipy matplotlib
```

### 1. Prepare training labels

```bash
python prepare_labels.py \
    --data data/aligned_training_data.npz \
    --out-dir data/processed
```

### 2. Train

```bash
python train.py \
    --data-dir data/processed \
    --epochs 100 \
    --stage-epochs 15 25 60 \
    --out-dir checkpoints
```

Training follows a three-stage curriculum:

```text
continuous latent learning
        ↓
structural-state learning
        ↓
full vector reconstruction
```

### 3. Evaluate

```bash
python eval.py \
    --data-dir data/processed \
    --ckpt checkpoints/final.pt \
    --out-dir results
```

### 4. Visualize

```bash
python visualize_reconstruction.py \
    --data-dir data/processed \
    --ckpt checkpoints/final.pt \
    --idx 1000
```

---

## Deep Learning Pipeline

```text
Scalar Measurement
        │
        ▼
Multi-View Feature Extraction
        │
        ├── Temporal Features
        ├── Local Features
        └── Time-Frequency Features
        │
        ▼
Gated Feature Fusion
        │
        ▼
Deep Shared Representation
        │
        ├── Continuous Latent Prediction
        ├── Discrete State Prediction
        └── Confidence Prediction
        │
        ▼
Structured Reconstruction
        │
        ▼
High-Dimensional Vector Signal
```

**NeuroUnfold turns scalar sensing into vector sensing by learning the latent structure that connects low-dimensional observations to high-dimensional signal representations.**

