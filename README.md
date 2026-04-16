# SELENE Engine Neural

A neural-controlled multi-agent visualization running on a wrapped toroidal state space.

This project turns the original SELENE manifold / lattice art concept into a small experimental sandbox where agents move on a periodic manifold, follow a leader, transfer leadership through proximity, and render their motion as a living 3D torus with trails.

## What it is

`selene_engine_neural.py` is a lightweight simulation with:

- a **toroidally wrapped state space**
- a **neural policy** that outputs motion updates
- **leader / follower dynamics**
- **tag-based leader handoff**
- **3D torus embedding**
- **animated trail rendering**

It is best understood as:

- an intuition tool for manifold-style dynamics
- a creative AI playground
- a starter scaffold for neural control experiments

## Current status

This version is **neural-controlled**, but the neural network is **not trained yet**.

That means:

- the model architecture is real
- the agents are driven by neural outputs
- behavior is currently random / untrained
- this is a foundation for training, imitation, or RL

## File

- `selene_engine_neural.py` — neural manifold simulation and renderer

## Requirements

Install the basic dependencies first:

```bash
pip install numpy matplotlib torch
```

## Run

Launch the simulation:

```bash
python selene_engine_neural.py
```

This opens a matplotlib animation window showing the shards moving across the toroidal embedding.

## How it works

### 1. Wrapped manifold state

Each shard lives in a 2D wrapped coordinate system:

- `u` in `[0, 1)` — toroidal loop coordinate
- `v` in `[0, 2π)` — angular coordinate

These coordinates are periodic, so motion wraps around instead of hitting edges.

### 2. Neural control

Each shard feeds a small observation into the neural policy:

- its own wrapped position
- wrapped delta to the leader

The policy outputs a 2D acceleration-like update.

In the current code, the network input is:

```python
[self.pos[0], self.pos[1], du, dv]
```

and the model outputs:

```python
acc = policy(inp)
```

### 3. Leader / follower dynamics

One shard is marked as the leader.

Other shards move according to the neural policy relative to that leader.

When a shard gets close enough, it becomes the new leader.

This creates a simple multi-agent handoff loop.

### 4. Toroidal embedding

The wrapped manifold state is mapped into 3D using a torus transform.

This means the agents do not just move in flat space. Their latent / state coordinates are embedded into a curved visible surface.

## Why this is useful

This file helps as a small sandbox for:

- manifold intuition
- multi-agent behavior
- neural control experiments
- MARL prototyping
- creative scientific visualization

It bridges:

- geometry
- art
- agent dynamics
- neural policy structure

## Limitations

This is **not** yet:

- a trained RL system
- a benchmark environment
- a full graph-neural controller
- a full Riemannian solver

It is a clean prototype and extension point.

## Suggested next upgrades

### Train the policy
Add a reward function and optimize the neural policy so followers learn to:

- approach the leader
- avoid jitter
- maintain smooth motion

### Add neighbor awareness
Instead of only using leader-relative input, include nearby shards.

That would allow:

- local coordination
- graph-like behavior
- richer emergence

### Add memory
Swap the simple MLP for:

- an RNN
- a GRU
- a small transformer block

This would let each shard use temporal context.

### Add dataset logging
Save trajectories, states, and rendered frames to disk for:

- world-model training
- imitation learning
- visualization archives

## Grounded framing

The code is best described as:

> a neural-controlled multi-agent toy world on a wrapped toroidal manifold

That framing is accurate and useful.

## Example extension direction

A natural next version would:

1. keep the toroidal state space
2. keep the 3D torus rendering
3. replace random neural behavior with learned policies
4. optionally add rewards and a training loop

## License / use

Use it freely as a personal experiment scaffold, visualization toy, or prototype base. If you publish an evolved version, it helps to document what changed in the dynamics and training setup.

## Quick summary

SELENE Engine Neural is a compact experimental file where:

- geometry is curved
- agents are neural-driven
- leadership can transfer
- motion leaves memory trails
- the result is both visual and structurally extensible
