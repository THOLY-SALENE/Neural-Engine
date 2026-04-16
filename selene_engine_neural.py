
"""
SELENE ENGINE v5 — NEURAL AGENT VERSION

Now includes:
- Neural policy (PyTorch)
- Multi-agent control via learned behavior
- Ready for RL / imitation / experimentation

Run:
python selene_engine_neural.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==========================
# CONFIG
# ==========================

NUM_SHARDS = 8
TRAIL_LENGTH = 80
TAG_DISTANCE = 0.2

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

    def forward(self, x):
        return self.net(x)

policy = NeuralPolicy().to(DEVICE)

# ==========================
# AGENT
# ==========================

class Shard:
    def __init__(self, i):
        self.id = i
        self.pos = np.array([random.random(), random.uniform(0, 2*np.pi)])
        self.vel = np.zeros(2)
        self.is_it = False

    def step(self, leader):
        du = leader.pos[0] - self.pos[0]
        if abs(du) > 0.5:
            du -= np.sign(du)

        dv = np.arctan2(
            np.sin(leader.pos[1] - self.pos[1]),
            np.cos(leader.pos[1] - self.pos[1])
        )

        inp = torch.tensor([self.pos[0], self.pos[1], du, dv], dtype=torch.float32)
        acc = policy(inp).detach().numpy()

        self.vel = self.vel * 0.85 + acc * 0.1
        self.pos += self.vel

        self.pos[0] %= 1.0
        self.pos[1] %= 2*np.pi


# ==========================
# TORUS
# ==========================

def torus(u, v, t):
    R = 4 + 0.5*np.sin(t)
    r = 1.5 + 0.2*np.cos(v+t)

    theta = u * 2*np.pi
    phi = v

    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r*np.sin(phi)

    return x, y, z


# ==========================
# ENGINE
# ==========================

class Engine:
    def __init__(self):
        self.shards = [Shard(i) for i in range(NUM_SHARDS)]
        self.leader = random.choice(self.shards)
        self.leader.is_it = True
        self.trails = [[] for _ in self.shards]

    def step(self, t):
        for s in self.shards:
            s.step(self.leader)

        for s in self.shards:
            if s is self.leader:
                continue
            if np.linalg.norm(self.leader.pos - s.pos) < TAG_DISTANCE:
                self.leader.is_it = False
                s.is_it = True
                self.leader = s
                break

    def positions(self, t):
        pts = []
        for i, s in enumerate(self.shards):
            xyz = torus(s.pos[0], s.pos[1], t)
            self.trails[i].append(xyz)
            if len(self.trails[i]) > TRAIL_LENGTH:
                self.trails[i].pop(0)
            pts.append(xyz)
        return pts


# ==========================
# VISUAL
# ==========================

def run():
    engine = Engine()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.axis("off")

        t = frame * 0.05
        engine.step(t)
        pts = engine.positions(t)

        for i, s in enumerate(engine.shards):
            x,y,z = pts[i]

            if len(engine.trails[i]) > 2:
                trail = np.array(engine.trails[i])
                ax.plot(trail[:,0], trail[:,1], trail[:,2], alpha=0.6)

            ax.scatter(x,y,z, s=150 if s.is_it else 50)

        ax.set_xlim(-6,6)
        ax.set_ylim(-6,6)
        ax.set_zlim(-6,6)

    FuncAnimation(fig, update, frames=400, interval=30)
    plt.show()


if __name__ == "__main__":
    run()
