# QMIX-NN Training Design: Alternative C

This note documents why the project moved to **Alternative C** for neural QMIX training and what trade-offs it addresses.

## Background

The earlier neural QMIX loop kept environment/control logic in Python + NumPy, while network calls used PyTorch internally. That design was simple and compatible with the existing runners, but it was still an **online, one-step-at-a-time** update pattern.

For deep RL, especially when targeting GPU acceleration, the strongest performance and stability gains typically come from:

1. **Replay buffering** (decorrelated training data)
2. **Mini-batch optimization** (better hardware utilization)
3. **Target networks** (stabilized TD targets)

That is exactly what Alternative C adds.

---

## Why Alternative C is usually better for deep QMIX

### 1) Replay buffer

Without replay, updates are strongly correlated in time, which can destabilize deep value learning. Replay enables random sampling across older and newer transitions.

Practical effect:
- More stable gradients
- Better sample reuse
- Less sensitivity to short-term trajectory noise

### 2) Mini-batch updates

Single-sample updates underutilize GPUs and are noisy. Mini-batches improve throughput and gradient quality.

Practical effect:
- Better compute efficiency on both CPU and GPU
- Smoother optimization
- More predictable learning behavior

### 3) Target networks

Bootstrapped TD targets can drift if computed from the same network being updated. Target networks provide a slowly-changing reference.

Practical effect:
- Reduced training oscillation/divergence
- More robust learning over long runs

---

## What changed conceptually

From:
- online one-step TD updates, no replay, no target network

To:
- transitions stored in replay buffer
- training starts after warmup (`learning_starts`)
- periodic batch updates (`batch_size`, `train_every`)
- target networks synchronized either:
  - **hard** every N gradient steps (`target_update_interval`), or
  - **soft** each update (`tau`)

---

## GPU implications

Alternative C is the first option that gives meaningful GPU leverage in this project context because it increases tensor batch sizes and reduces per-step tiny-kernel overhead.

- **Tiny online updates**: often bottlenecked by Python + kernel launch overhead
- **Batched replay updates**: better occupancy and throughput

Note: absolute speedup still depends on environment step cost and model size. If environment stepping dominates runtime, gains may be moderate.

---

## Configuration knobs (QMIX-NN)

In `agents/QMIXLearningNN/config/learning-params.json`:

- `replay_capacity`
- `batch_size`
- `learning_starts`
- `train_every`
- `target_update_mode` (`hard` or `soft`)
- `target_update_interval` (hard mode)
- `tau` (soft mode)
- `device` (`auto`, `cpu`, `cuda`, ...)

---

## Remaining constraints

This implementation still follows the current project runner/control style and environment API. It does not yet add advanced RL extras such as:

- prioritized replay
- n-step returns
- double-Q action selection in target computation
- sequence-based recurrent QMIX variants

Those can be layered on top if needed.

---

## Summary

Alternative C was chosen because it is the most practical path to both:

1. **better deep-RL training stability**, and
2. **real GPU-usable batching behavior**

while preserving compatibility with existing experiment orchestration and CLI patterns.

