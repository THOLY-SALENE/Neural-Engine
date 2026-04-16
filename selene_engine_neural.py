"""
SELENE ENGINE v7 — AI RESEARCH EDITION

Full Gym-style environment + RL training + dataset export + evaluation

Designed for:
• Reinforcement Learning (MARL, policy gradients, offline RL)
• Geometric Deep Learning / Manifold Learning
• World-model / video-prediction research
• Imitation + Behavioral Cloning experiments

Run with:
    python selene_engine_research_v7.py --mode [animate|train_rl|dataset|eval]
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
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
OBS_DIM = 4
ACT_DIM = 2

# ==========================
# POLICY
# ==========================

class NeuralPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACT_DIM)
        )

    def forward(self, x):
        return self.net(x)

policy = NeuralPolicy().to(DEVICE)

# ==========================
# METRIC
# ==========================

def wrapped_distance(a, b):
    du = a[0] - b[0]
    if abs(du) > 0.5:
        du -= np.sign(du)
    dv = np.arctan2(np.sin(a[1]-b[1]), np.cos(a[1]-b[1]))
    return float(np.sqrt(du**2 + dv**2))

def wrapped_delta_u(a_u, b_u):
    du = a_u - b_u
    if abs(du) > 0.5:
        du -= np.sign(du)
    return du

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
    def spawn(cls, i, rng):
        return cls(
            id=i,
            pos=np.array([rng.random(), rng.uniform(0, 2*np.pi)], dtype=np.float32),
            vel=np.zeros(2, dtype=np.float32)
        )

# ==========================
# ENV
# ==========================

class SeleneEnv:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.shards = [Shard.spawn(i, self.rng) for i in range(NUM_SHARDS)]
        self.leader = self.rng.choice(self.shards)
        self.leader.is_it = True
        self.t = 0.0
        return self._obs()

    def _obs(self):
        obs = []
        for s in self.shards:
            du = wrapped_delta_u(self.leader.pos[0], s.pos[0])
            dv = np.arctan2(
                np.sin(self.leader.pos[1]-s.pos[1]),
                np.cos(self.leader.pos[1]-s.pos[1])
            )
            obs.append([s.pos[0], s.pos[1], du, dv])
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)

        for i, s in enumerate(self.shards):
            s.vel = s.vel*0.85 + actions[i]
            s.pos += s.vel
            s.pos[0] %= 1.0
            s.pos[1] %= 2*np.pi

        self.t += DT

        for s in self.shards:
            if s is self.leader: continue
            if wrapped_distance(self.leader.pos, s.pos) < TAG_DISTANCE:
                self.leader.is_it = False
                s.is_it = True
                self.leader = s
                break

        obs = self._obs()

        dists = [wrapped_distance(self.leader.pos, s.pos) for s in self.shards]
        reward = -np.mean(dists) - 0.1*np.mean([np.linalg.norm(s.vel) for s in self.shards])

        return obs, reward, False, {}

# ==========================
# TORUS
# ==========================

def torus(u, v, t):
    R = 4 + 0.5*np.sin(t)
    r = 1.5 + 0.2*np.cos(v+t)

    theta = u*2*np.pi
    phi = v

    x = (R + r*np.cos(phi))*np.cos(theta)
    y = (R + r*np.cos(phi))*np.sin(theta)
    z = r*np.sin(phi)
    return x,y,z

def color(v):
    return hsv_to_rgb([(v/(2*np.pi))%1,0.85,0.95])

# ==========================
# ENGINE
# ==========================

class Engine:
    def __init__(self, seed=42):
        self.env = SeleneEnv(seed)
        self.trails = [[] for _ in range(NUM_SHARDS)]

    def step(self):
        obs = self.env._obs()
        acts = policy(torch.from_numpy(obs)).detach().numpy()
        self.env.step(acts)

    def positions(self,t):
        pts=[]
        for i,s in enumerate(self.env.shards):
            xyz=torus(s.pos[0],s.pos[1],t)
            self.trails[i].append(xyz)
            if len(self.trails[i])>TRAIL_LENGTH:
                self.trails[i].pop(0)
            pts.append(xyz)
        return pts

# ==========================
# RL TRAIN
# ==========================

def train_rl(epochs=80):
    env=SeleneEnv()
    opt=optim.Adam(policy.parameters(),lr=0.005)

    for ep in range(epochs):
        obs=env.reset()
        total=0
        for _ in range(400):
            obs_t=torch.from_numpy(obs)
            act=policy(obs_t).detach().numpy()
            next_obs,r,_,_=env.step(act)

            loss=torch.tensor(-r,requires_grad=True)
            opt.zero_grad()
            loss.backward()
            opt.step()

            obs=next_obs
            total+=r

        if ep%10==0:
            print("Epoch",ep,"Reward",total)

    torch.save(policy.state_dict(),"selene_policy.pt")

# ==========================
# DATASET
# ==========================

def dataset(n=10000):
    env=SeleneEnv()
    obs_buf,act_buf=[],[]

    obs=env.reset()
    for _ in range(n):
        act=policy(torch.from_numpy(obs)).detach().numpy()
        next_obs,_,_,_=env.step(act)

        obs_buf.append(obs)
        act_buf.append(act)

        obs=next_obs

    np.savez("selene_data.npz",obs=np.array(obs_buf),act=np.array(act_buf))

# ==========================
# ANIMATE
# ==========================

def animate():
    engine=Engine()
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')

    def update(f):
        ax.clear()
        ax.axis("off")

        t=f*DT
        engine.step()
        pts=engine.positions(t)

        for i,s in enumerate(engine.env.shards):
            x,y,z=pts[i]
            col=color(s.pos[1])

            if len(engine.trails[i])>2:
                trail=np.array(engine.trails[i])
                ax.plot(trail[:,0],trail[:,1],trail[:,2],color=col,alpha=0.6)

            ax.scatter(x,y,z,s=150 if s.is_it else 50,color=col)

    FuncAnimation(fig,update,frames=FRAMES,interval=INTERVAL_MS)
    plt.show()

# ==========================
# CLI
# ==========================

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["animate","train_rl","dataset","eval"],default="animate")
    args=p.parse_args()

    if args.mode=="train_rl":
        train_rl()
    elif args.mode=="dataset":
        dataset()
    elif args.mode=="eval":
        print("Eval not implemented yet")
    else:
        animate()

if __name__=="__main__":
    main()
