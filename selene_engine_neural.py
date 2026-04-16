"""
SELENE ENGINE v53 — FULL Y E E T COGNITIVE TORUS STACK
Everything: Beautiful pulsing visualization + Stable PPO + Transformer Memory + 
Communication + Latent World Model + Lightweight Planning + Dataset + Eval

Run:
  python selene_engine_v53.py --mode train --epochs 100
  python selene_engine_v53.py --mode animate
  python selene_engine_v53.py --mode dataset
  python selene_engine_v53.py --mode eval
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
NUM_ENVS = 4
SEQ_LEN = 8
ROLLOUT = 128
PPO_EPOCHS = 4
MINIBATCH = 128
GAMMA = 0.99
CLIP = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= WORLD MODEL =================
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


world_model = WorldModel().to(DEVICE)


# ================= MEMORY =================
class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, seq):  # (B, T, N, F)
        B, T, N, F = seq.shape
        x = seq.view(B * N, T, F)
        x = self.embed(x)
        x = self.tr(x)
        return x[:, -1].view(B, N, 64)


# ================= COMM =================
class Comm(nn.Module):
    def __init__(self):
        super().__init__()
        self.msg = nn.Linear(64, 32)

    def forward(self, h):  # (B, N, 64)
        return self.msg(h).mean(dim=1, keepdim=True).repeat(1, h.shape[1], 1)


# ================= POLICY =================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = Memory()
        self.comm = Comm()
        self.head = nn.Sequential(
            nn.Linear(96, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.mean = nn.Linear(128, 2)
        self.value = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, seq):  # (B, T, N, 4)
        h = self.mem(seq)
        c = self.comm(h)
        x = torch.cat([h, c], dim=-1)
        x = self.head(x)
        mean = self.mean(x)
        value = self.value(x).mean(dim=1)
        return mean, value


policy = Policy().to(DEVICE)


# ================= ENV =================
def wrap(a, b):
    d = a - b
    if abs(d) > 0.5:
        d -= np.sign(d)
    return d


def dist(a, b):
    du = wrap(a[0], b[0])
    dv = np.arctan2(np.sin(a[1] - b[1]), np.cos(a[1] - b[1]))
    return np.sqrt(du**2 + dv**2)


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
            du = wrap(self.pos[self.leader][0], self.pos[i][0])
            dv = np.arctan2(np.sin(self.pos[self.leader][1] - self.pos[i][1]),
                            np.cos(self.pos[self.leader][1] - self.pos[i][1]))
            o.append([self.pos[i][0], self.pos[i][1], du, dv])
        return np.array(o, dtype=np.float32)

    def step(self, act):
        act = np.asarray(act, dtype=np.float32).reshape(NUM_SHARDS, 2)
        self.vel = self.vel * 0.85 + act
        self.pos += self.vel
        self.pos[:, 0] %= 1.0
        self.pos[:, 1] %= 2 * np.pi

        handoff = False
        for i in range(NUM_SHARDS):
            if i != self.leader and dist(self.pos[self.leader], self.pos[i]) < 0.2:
                self.leader = i
                handoff = True
                break

        avg_dist = np.mean([dist(self.pos[self.leader], self.pos[i]) for i in range(NUM_SHARDS)])
        reward = -avg_dist + (15.0 if handoff else 0.0)

        return self.obs(), reward, False, {"handoff": handoff}


# ================= PLANNING =================
def plan(obs):
    s = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    best_a, best_score = None, -1e9
    for _ in range(6):
        a = torch.randn(NUM_SHARDS, 2, device=DEVICE) * 0.12
        pred = world_model(s, a)
        score = -((pred[:, 2:] ** 2).mean())
        if score > best_score:
            best_score = score
            best_a = a
    return best_a.cpu().numpy()


# ================= TRAINING =================
def train(epochs=80):
    print("=== SELENE v53 FULL Y E E T TRAINING ===")
    opt = optim.Adam(policy.parameters(), lr=3e-4)
    wm_opt = optim.Adam(world_model.parameters(), lr=3e-4)

    for ep in range(epochs):
        envs = [Env(42 + i) for i in range(NUM_ENVS)]
        obs_list = [env.reset() for env in envs]
        seqs = [[obs.copy() for _ in range(SEQ_LEN)] for obs in obs_list]

        obs_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []

        for _ in range(ROLLOUT):
            seq_batch = torch.tensor([np.array(s[-SEQ_LEN:]) for s in seqs], dtype=torch.float32, device=DEVICE)

            mean, val = policy(seq_batch)
            distn = torch.distributions.Normal(mean, torch.exp(policy.log_std))
            act = distn.sample()

            next_obs_list, rewards = [], []
            for i, env in enumerate(envs):
                planned = plan(obs_list[i])
                blended = 0.65 * act[i].cpu().numpy() + 0.35 * planned

                next_obs, r, _, _ = env.step(blended)
                next_obs_list.append(next_obs)
                rewards.append(r)

                # World model update
                s_t = torch.tensor(obs_list[i], dtype=torch.float32, device=DEVICE)
                a_t = torch.tensor(blended, dtype=torch.float32, device=DEVICE)
                target = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
                wm_loss = ((world_model(s_t, a_t) - target)**2).mean()
                wm_opt.zero_grad()
                wm_loss.backward()
                wm_opt.step()

            obs_buf.append(seq_batch)
            act_buf.append(act)
            logp_buf.append(distn.log_prob(act).sum(-1).mean(dim=1))
            val_buf.append(val)
            rew_buf.append(torch.tensor(rewards, device=DEVICE))

            obs_list = next_obs_list
            for i in range(NUM_ENVS):
                seqs[i].append(obs_list[i])

        # GAE
        returns = []
        G = torch.zeros(NUM_ENVS, device=DEVICE)
        for r in reversed(rew_buf):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.stack(returns)

        values = torch.stack(val_buf)
        adv = (returns - values.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_flat = torch.cat(obs_buf)
        act_flat = torch.cat(act_buf)
        logp_old = torch.cat(logp_buf).detach()
        adv_flat = adv.view(-1)
        ret_flat = returns.view(-1)

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(len(obs_flat))
            for start in range(0, len(idx), MINIBATCH):
                mb = idx[start:start + MINIBATCH]

                mean, val = policy(obs_flat[mb])
                distn = torch.distributions.Normal(mean, torch.exp(policy.log_std))
                logp = distn.log_prob(act_flat[mb]).sum(-1).mean(dim=1)

                ratio = torch.exp(logp - logp_old[mb])
                s1 = ratio * adv_flat[mb]
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_flat[mb]

                policy_loss = -torch.min(s1, s2).mean()
                value_loss = 0.5 * (val - ret_flat[mb]).pow(2).mean()

                loss = policy_loss + value_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

        print(f"Epoch {ep:3d} | Avg Reward {rew_buf[-1].mean().item():.2f}")

    torch.save(policy.state_dict(), "selene_v53.pt")
    print("Training complete → selene_v53.pt\n")


# ================= BEAUTIFUL VISUALIZATION =================
def torus(u, v, t):
    R = 4 + 0.5 * np.sin(t)
    r = 1.5 + 0.2 * np.cos(v + t)
    th = u * 2 * np.pi
    x = (R + r * np.cos(v)) * np.cos(th)
    y = (R + r * np.cos(v)) * np.sin(th)
    z = r * np.sin(v)
    return x, y, z


def animate():
    print("Launching the beautiful pulsing torus...")
    if Path("selene_v53.pt").exists():
        policy.load_state_dict(torch.load("selene_v53.pt", map_location=DEVICE))
        print("Loaded trained policy.")

    env = Env(42)
    trails = [[] for _ in range(NUM_SHARDS)]

    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")

    def update(f):
        ax.clear()
        ax.axis("off")
        ax.set_facecolor((0.0, 0.0, 0.05))

        t = f * 0.05
        obs = env.obs()
        seq = np.array([obs] * SEQ_LEN)
        seq_t = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            mean, _ = policy(seq_t)
            act = mean[0].cpu().numpy()

        env.step(act)

        for i in range(NUM_SHARDS):
            u, v = env.pos[i]
            x, y, z = torus(u, v, t)
            trails[i].append((x, y, z))
            if len(trails[i]) > 80:
                trails[i].pop(0)

            col = hsv_to_rgb([(v / (2 * np.pi)) % 1, 0.85, 0.95])
            pts = np.array(trails[i])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, alpha=0.65, lw=1.8)

            size = 180 if i == env.leader else 55
            ax.scatter(x, y, z, s=size,
                       color="white" if i == env.leader else col,
                       edgecolors="white",
                       linewidth=2.2 if i == env.leader else 0,
                       alpha=0.95)

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        ax.set_title("SELENE v53 — Full Cognitive Torus (Memory + Comm + Planning)", color="cyan", fontsize=12)

    FuncAnimation(fig, update, frames=400, interval=30)
    plt.show()


# ================= DATASET EXPORT =================
def generate_dataset(num_steps=10000, output="selene_dataset.npz"):
    print(f"Generating dataset ({num_steps} steps)...")
    env = Env(123)
    obs_l, act_l, rew_l, next_l, done_l = [], [], [], [], []

    obs = env.reset()
    for _ in range(num_steps):
        seq = np.array([obs] * SEQ_LEN)
        seq_t = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            mean, _ = policy(seq_t)
            act = mean[0].cpu().numpy()

        next_obs, r, done, _ = env.step(act)

        obs_l.append(obs)
        act_l.append(act)
        rew_l.append(r)
        next_l.append(next_obs)
        done_l.append(done)

        obs = next_obs if not done else env.reset()

    np.savez_compressed(output, obs=np.array(obs_l), act=np.array(act_l),
                        rew=np.array(rew_l), next_obs=np.array(next_l), done=np.array(done_l))
    print(f"Dataset saved → {output}")


# ================= EVAL =================
def evaluate(steps=400):
    print("Running evaluation...")
    if Path("selene_v53.pt").exists():
        policy.load_state_dict(torch.load("selene_v53.pt", map_location=DEVICE))
    env = Env(42)
    obs = env.reset()
    total_r = 0.0
    handoffs = 0
    for _ in range(steps):
        seq = np.array([obs] * SEQ_LEN)
        seq_t = torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean, _ = policy(seq_t)
            act = mean[0].cpu().numpy()
        obs, r, _, info = env.step(act)
        total_r += r
        if info.get("handoff"):
            handoffs += 1
    print(f"Eval → Total Reward: {total_r:.2f} | Handoffs: {handoffs} | Avg/step: {total_r/steps:.3f}")


# ================= CLI =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SELENE ENGINE v53 — Full Yeet Cognitive Torus")
    parser.add_argument("--mode", choices=["train", "animate", "dataset", "eval"], default="animate")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs)
    elif args.mode == "dataset":
        generate_dataset()
    elif args.mode == "eval":
        evaluate()
    else:
        animate()
