"""
SELENE ENGINE v14 — STABLE FULL RESEARCH STACK

Based on your v13.1 with fixed GAE shapes + enhanced reward + visuals.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb

# ================= CONFIG =================
NUM_SHARDS = 8
NUM_ENVS = 8
ROLLOUT = 256
TAG_DISTANCE = 0.2
DT = 0.05
GAMMA = 0.99
LAMBDA = 0.95
DEVICE = "cpu"

# ================= POLICY =================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(4,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh()
        )
        self.mean = nn.Linear(128,2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.value = nn.Linear(128,1)

    def forward(self,x):
        h = self.base(x)
        return self.mean(h), torch.exp(self.log_std), self.value(h)

policy = Policy().to(DEVICE)

# ================= METRIC & ENV =================
def wrap_du(a, b):
    d = a - b
    if abs(d) > 0.5:
        d -= np.sign(d)
    return float(d)

def dist(a, b):
    du = wrap_du(a[0], b[0])
    dv = np.arctan2(np.sin(a[1]-b[1]), np.cos(a[1]-b[1]))
    return float(np.sqrt(du**2 + dv**2))

class Env:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pos = np.array([[self.rng.random(), self.rng.uniform(0, 2*np.pi)] for _ in range(NUM_SHARDS)], dtype=np.float32)
        self.vel = np.zeros((NUM_SHARDS, 2), dtype=np.float32)
        self.leader = self.rng.randrange(NUM_SHARDS)
        return self.obs()

    def obs(self):
        o = []
        for i in range(NUM_SHARDS):
            du = wrap_du(self.pos[self.leader][0], self.pos[i][0])
            dv = np.arctan2(np.sin(self.pos[self.leader][1] - self.pos[i][1]),
                            np.cos(self.pos[self.leader][1] - self.pos[i][1]))
            o.append([self.pos[i][0], self.pos[i][1], du, dv])
        return np.array(o, dtype=np.float32)

    def step(self, act):
        act = np.asarray(act, dtype=np.float32).reshape(NUM_SHARDS, 2)
        self.vel = self.vel * 0.85 + act
        self.pos += self.vel
        self.pos[:,0] %= 1.0
        self.pos[:,1] %= 2*np.pi

        handoff = False
        for i in range(NUM_SHARDS):
            if i != self.leader and dist(self.pos[self.leader], self.pos[i]) < TAG_DISTANCE:
                self.leader = i
                handoff = True
                break

        avg_dist = np.mean([dist(self.pos[self.leader], self.pos[i]) for i in range(NUM_SHARDS)])
        energy = np.mean(np.linalg.norm(self.vel, axis=1))
        reward = -avg_dist + (12.0 if handoff else 0.0) - 0.08 * energy

        return self.obs(), reward, False, {"handoff": handoff}

# ================= TRAINING (Stable PPO) =================
def train(epochs=150, seed=42):
    print("=== SELENE v14 Training ===")
    envs = [Env(seed + i) for i in range(NUM_ENVS)]
    opt = optim.Adam(policy.parameters(), lr=3e-4)

    for ep in range(epochs):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []
        obs = [env.reset() for env in envs]

        for _ in range(ROLLOUT):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            mean, std, val = policy(obs_t)
            distn = torch.distributions.Normal(mean, std)
            act = distn.sample()
            logp = distn.log_prob(act).sum(-1)

            next_obs, rewards = [], []
            for i, env in enumerate(envs):
                o, r, _, _ = env.step(act[i].detach().cpu().numpy())
                next_obs.append(o)
                rewards.append(r)

            obs_buf.append(obs_t)
            act_buf.append(act)
            logp_buf.append(logp)
            val_buf.append(val.squeeze(-1))
            rew_buf.append(torch.tensor(rewards, device=DEVICE))

            obs = next_obs

        # Bootstrap
        last_val = policy(torch.tensor(obs, dtype=torch.float32, device=DEVICE))[2].squeeze(-1)
        val_buf.append(last_val)

        val_buf = torch.stack(val_buf)
        rew_buf = torch.stack(rew_buf)

        # GAE (T, N)
        adv = torch.zeros_like(rew_buf)
        gae = torch.zeros(NUM_ENVS, device=DEVICE)
        for t in reversed(range(ROLLOUT)):
            delta = rew_buf[t] + GAMMA * val_buf[t+1] - val_buf[t]
            gae = delta + GAMMA * LAMBDA * gae
            adv[t] = gae

        returns = adv + val_buf[:-1]

        # Flatten
        obs_flat = torch.cat(obs_buf)
        act_flat = torch.cat(act_buf)
        logp_old = torch.cat(logp_buf).detach()
        adv_flat = adv.reshape(-1)
        returns_flat = returns.reshape(-1)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for _ in range(4):
            mean, std, val = policy(obs_flat)
            distn = torch.distributions.Normal(mean, std)
            logp = distn.log_prob(act_flat).sum(-1)

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * ((val.squeeze(-1) - returns_flat)**2).mean()
            entropy = distn.entropy().mean()

            loss = policy_loss + value_loss - 0.01 * entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % 20 == 0 or ep == epochs - 1:
            avg_r = rew_buf[-1].mean().item()
            print(f"Epoch {ep:3d} | Avg Reward: {avg_r:.2f}")

    torch.save(policy.state_dict(), "selene_v14.pt")
    print("Training complete → selene_v14.pt\n")

# ================= VISUAL (Beautiful) =================
def torus(u, v, t):
    R = 4 + 0.5 * np.sin(t)
    r = 1.5 + 0.2 * np.cos(v + t)
    th = u * 2 * np.pi
    x = (R + r * np.cos(v)) * np.cos(th)
    y = (R + r * np.cos(v)) * np.sin(th)
    z = r * np.sin(v)
    return x, y, z

def animate(seed=42):
    print("Launching animation...")
    if Path("selene_v14.pt").exists():
        policy.load_state_dict(torch.load("selene_v14.pt", map_location=DEVICE))
    env = Env(seed)
    trails = [[] for _ in range(NUM_SHARDS)]

    fig = plt.figure(figsize=(10,10), facecolor="black")
    ax = fig.add_subplot(111, projection='3d')

    def update(f):
        ax.clear()
        ax.axis("off")
        ax.set_facecolor((0,0,0.05))

        t = f * DT
        obs = env.obs()
        with torch.no_grad():
            mean,_,_ = policy(torch.tensor(obs, dtype=torch.float32, device=DEVICE))
            act = mean.cpu().numpy()
        env.step(act)

        for i in range(NUM_SHARDS):
            x,y,z = torus(env.pos[i][0], env.pos[i][1], t)
            trails[i].append((x,y,z))
            if len(trails[i]) > 80: trails[i].pop(0)
            pts = np.array(trails[i])
            col = hsv_to_rgb([(env.pos[i][1]/(2*np.pi))%1, 0.85, 0.95])
            ax.plot(pts[:,0], pts[:,1], pts[:,2], color=col, alpha=0.65)
            ax.scatter(x,y,z, s=180 if i==env.leader else 55, color=col)

        ax.set_xlim(-6,6)
        ax.set_ylim(-6,6)
        ax.set_zlim(-6,6)

    FuncAnimation(fig, update, frames=400, interval=30)
    plt.show()

# ================= CLI =================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "animate"], default="animate")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs, seed=args.seed)
    else:
        animate(seed=args.seed)
