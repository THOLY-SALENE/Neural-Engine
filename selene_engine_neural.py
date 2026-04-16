"""
SELENE ENGINE v6 — NEURAL AGENT + TRAINING LOOP

Upgrades from v5:
- Fixed toroidal wrapped distance (tag handoffs now work correctly)
- Full imitation-learning training loop (behavioral cloning from expert dynamics)
- Policy learns smooth pursuit + leader exploration on CPU
- Per-shard colors + trail rendering
- After training, the animation runs with the learned neural policy

The neural agents now behave like the earlier hand-coded version —
but are driven by the trained network after the imitation phase.

Run:
    python selene_engine_neural_v6.py

Optional:
    python selene_engine_neural_v6.py --epochs 60
    python selene_engine_neural_v6.py --seed 123
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb


# ==========================
# CONFIG
# ==========================

NUM_SHARDS = 8
TRAIL_LENGTH = 80
TAG_DISTANCE = 0.2
FRAMES = 400
INTERVAL_MS = 30
DT = 0.05

DEVICE = "cpu"


# ==========================
# NEURAL POLICY
# ==========================

class NeuralPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


policy = NeuralPolicy().to(DEVICE)


# ==========================
# METRIC
# ==========================

def wrapped_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Toroidal/wrapped distance in latent state space."""
    du = a[0] - b[0]
    if abs(du) > 0.5:
        du -= np.sign(du)

    dv = np.arctan2(np.sin(a[1] - b[1]), np.cos(a[1] - b[1]))
    return float(np.sqrt(du**2 + dv**2))


def wrapped_delta_u(a_u: float, b_u: float) -> float:
    du = a_u - b_u
    if abs(du) > 0.5:
        du -= np.sign(du)
    return float(du)


# ==========================
# AGENT
# ==========================

@dataclass
class Shard:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    is_it: bool = False

    @classmethod
    def spawn(cls, i: int, rng: random.Random) -> "Shard":
        return cls(
            id=i,
            pos=np.array([rng.random(), rng.uniform(0, 2 * np.pi)], dtype=np.float32),
            vel=np.zeros(2, dtype=np.float32),
            is_it=False,
        )

    def neural_step(self, leader: "Shard") -> None:
        """Neural forward pass → acceleration."""
        du = wrapped_delta_u(leader.pos[0], self.pos[0])
        dv = np.arctan2(
            np.sin(leader.pos[1] - self.pos[1]),
            np.cos(leader.pos[1] - self.pos[1])
        )

        inp = torch.tensor(
            [self.pos[0], self.pos[1], du, dv],
            dtype=torch.float32,
            device=DEVICE
        )
        acc = policy(inp).detach().cpu().numpy().astype(np.float32)

        self.vel = self.vel * 0.85 + acc
        self.pos += self.vel

        self.pos[0] %= 1.0
        self.pos[1] %= 2 * np.pi


# ==========================
# EXPERT DYNAMICS
# ==========================

def expert_acceleration(shard: Shard, leader: Shard, t: float) -> np.ndarray:
    """Hand-coded expert policy used for imitation learning."""
    du = wrapped_delta_u(leader.pos[0], shard.pos[0])
    dv = np.arctan2(
        np.sin(leader.pos[1] - shard.pos[1]),
        np.cos(leader.pos[1] - shard.pos[1])
    )

    if shard.is_it:
        target_acc = np.array([
            0.02 * np.sin(t + shard.pos[1]),
            0.02 * np.cos(t + shard.pos[0] * 2 * np.pi)
        ], dtype=np.float32)
    else:
        target_acc = np.array([du * 0.18, dv * 0.12], dtype=np.float32)

    return target_acc


# ==========================
# TORUS EMBEDDING
# ==========================

def torus(u: float, v: float, t: float) -> Tuple[float, float, float]:
    R = 4 + 0.5 * np.sin(t)
    r = 1.5 + 0.2 * np.cos(v + t)

    theta = u * 2 * np.pi
    phi = v

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    return float(x), float(y), float(z)


def color_from_angle(v: float) -> np.ndarray:
    hue = (v / (2 * np.pi)) % 1
    return hsv_to_rgb([hue, 0.85, 0.95])


# ==========================
# ENGINE
# ==========================

class Engine:
    def __init__(self, seed: int = 42):
        rng = random.Random(seed)
        self.shards: List[Shard] = [Shard.spawn(i, rng) for i in range(NUM_SHARDS)]
        self.leader = rng.choice(self.shards)
        self.leader.is_it = True
        self.trails: List[List[Tuple[float, float, float]]] = [[] for _ in self.shards]

    def step(self, t: float) -> None:
        for s in self.shards:
            s.neural_step(self.leader)

        for s in self.shards:
            if s is self.leader:
                continue
            if wrapped_distance(self.leader.pos, s.pos) < TAG_DISTANCE:
                self.leader.is_it = False
                s.is_it = True
                self.leader = s
                break

    def positions(self, t: float) -> List[Tuple[float, float, float]]:
        pts = []
        for i, s in enumerate(self.shards):
            xyz = torus(s.pos[0], s.pos[1], t)
            self.trails[i].append(xyz)
            if len(self.trails[i]) > TRAIL_LENGTH:
                self.trails[i].pop(0)
            pts.append(xyz)
        return pts


# ==========================
# TRAINING: IMITATION LEARNING
# ==========================

def generate_expert_data(num_steps: int = 800, seed: int = 123) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run a classic hand-coded simulation and collect (state, expert_acc) pairs.
    """
    rng = random.Random(seed)
    shards = [Shard.spawn(i, rng) for i in range(NUM_SHARDS)]
    leader = rng.choice(shards)
    leader.is_it = True

    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for step in range(num_steps):
        t = step * DT

        for s in shards:
            du = wrapped_delta_u(leader.pos[0], s.pos[0])
            dv = np.arctan2(
                np.sin(leader.pos[1] - s.pos[1]),
                np.cos(leader.pos[1] - s.pos[1])
            )

            inp = torch.tensor(
                [s.pos[0], s.pos[1], du, dv],
                dtype=torch.float32,
                device=DEVICE
            )

            target_acc = expert_acceleration(s, leader, t)

            inputs.append(inp)
            targets.append(torch.from_numpy(target_acc).to(DEVICE))

            vel = s.vel * 0.85 + target_acc
            s.pos += vel
            s.pos[0] %= 1.0
            s.pos[1] %= 2 * np.pi
            s.vel = vel

        for s in shards:
            if s is leader:
                continue
            if wrapped_distance(leader.pos, s.pos) < TAG_DISTANCE:
                leader.is_it = False
                s.is_it = True
                leader = s
                break

    return inputs, targets


def train_policy(num_epochs: int = 40, lr: float = 0.008, data_steps: int = 800, seed: int = 123) -> None:
    """Train the shared neural policy via behavioral cloning."""
    print("Collecting expert trajectories for imitation learning...")
    inputs, targets = generate_expert_data(num_steps=data_steps, seed=seed)
    print(f"Collected {len(inputs):,} training samples.")

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    batch_size = 64
    n = len(inputs)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        perm = torch.randperm(n)

        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_in = torch.stack([inputs[j] for j in batch_idx])
            batch_tgt = torch.stack([targets[j] for j in batch_idx])

            optimizer.zero_grad()
            pred = policy(batch_in)
            loss = criterion(pred, batch_tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, (n // batch_size) + 1)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch + 1:2d}/{num_epochs} — Avg MSE: {avg_loss:.6f}")

    print("Training complete. Neural policy now drives the simulation.\n")


# ==========================
# VISUALIZATION
# ==========================

def run(seed: int = 42, epochs: int = 40) -> None:
    train_policy(num_epochs=epochs, seed=seed + 1000)

    print("Launching live simulation with trained neural agents...")
    engine = Engine(seed=seed)

    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_facecolor((0.0, 0.0, 0.05))

        t = frame * DT
        engine.step(t)
        pts = engine.positions(t)

        for i, s in enumerate(engine.shards):
            x, y, z = pts[i]
            col = color_from_angle(s.pos[1])

            if len(engine.trails[i]) > 2:
                trail = np.array(engine.trails[i])
                ax.plot(
                    trail[:, 0], trail[:, 1], trail[:, 2],
                    color=col, alpha=0.65, lw=1.8
                )

            size = 180 if s.is_it else 55
            ax.scatter(
                x, y, z,
                s=size,
                color=col if not s.is_it else "white",
                edgecolors="white",
                linewidth=2.2 if s.is_it else 0,
                alpha=0.95
            )

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        ax.set_title(
            "SELENE ENGINE v6 — Neural Agents on Torus (trained)",
            color="cyan",
            fontsize=12
        )

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL_MS)
    plt.show()


# ==========================
# CLI
# ==========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SELENE ENGINE v6 — neural agents + imitation learning")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sim init")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(seed=args.seed, epochs=args.epochs)
