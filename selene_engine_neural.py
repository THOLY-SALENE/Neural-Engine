"""
SELENE ENGINE v75 — FEDERATED GOVERNANCE + TYPED VETO + NEIGHBOR COMM

Adds:
- Typed Black veto (soft / hard / critical)
- Federation voting system
- K-nearest neighbor communication
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ================= CONFIG =================
NUM_SHARDS = 32
SEQ_LEN = 8
K_NEIGHBORS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MEMORY =================
class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 64)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, seq):
        B,T,N,F = seq.shape
        x = seq.reshape(B*N, T, F)
        x = self.embed(x)
        x = self.tr(x)
        return x[:, -1].reshape(B, N, 64)

# ================= NEIGHBOR COMM =================
class NeighborComm(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64,64)
        self.k = nn.Linear(64,64)
        self.v = nn.Linear(64,64)

    def forward(self, h, pos):
        B,N,_ = h.shape

        # compute pairwise distances
        dists = torch.cdist(pos, pos)  # (B,N,N)

        # get K nearest indices
        knn_idx = dists.argsort(dim=-1)[:,:,:K_NEIGHBORS]

        Q,K,V = self.q(h), self.k(h), self.v(h)

        out = torch.zeros_like(h)

        for b in range(B):
            for i in range(N):
                idx = knn_idx[b,i]
                k = K[b,idx]
                v = V[b,idx]
                q = Q[b,i]

                attn = torch.softmax((q @ k.T)/np.sqrt(64), dim=-1)
                out[b,i] = attn @ v

        return out

# ================= MCP =================
class MCP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,h):
        return torch.sigmoid(self.net(h))

# ================= HEART (TYPED BLACK) =================
class Heart(nn.Module):
    def __init__(self):
        super().__init__()
        self.black_soft = nn.Linear(64,1)
        self.black_hard = nn.Linear(64,1)
        self.black_crit = nn.Linear(64,1)
        self.red = nn.Linear(64,1)
        self.green = nn.Linear(64,1)

    def forward(self,h):
        return {
            "black_soft": torch.sigmoid(self.black_soft(h)),
            "black_hard": torch.sigmoid(self.black_hard(h)),
            "black_crit": torch.sigmoid(self.black_crit(h)),
            "red": torch.sigmoid(self.red(h)),
            "green": torch.sigmoid(self.green(h)),
        }

# ================= FEDERATION VOTING =================
def federation_vote(black_crit):
    # black_crit: (B,N,1)
    votes = (black_crit > 0.7).float()
    vote_ratio = votes.mean()

    return vote_ratio > 0.4  # threshold

# ================= POLICY =================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = Memory()
        self.comm = NeighborComm()
        self.mcp = MCP()
        self.heart = Heart()

        self.head = nn.Sequential(
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )

        self.mean = nn.Linear(128,2)
        self.value = nn.Linear(128,1)
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, seq, pos):
        h = self.mem(seq)
        c = self.comm(h, pos)

        x = torch.cat([h,c], dim=-1)

        mcp = self.mcp(h)
        hearts = self.heart(h)

        x = x * mcp
        x = self.head(x)

        mean = self.mean(x)
        value = self.value(x).mean(dim=1)

        return mean, value, hearts

policy = Policy().to(DEVICE)

# ================= ENV =================
def wrap(a,b):
    d = a - b
    if abs(d)>0.5: d -= np.sign(d)
    return d

def dist(a,b):
    du = wrap(a[0],b[0])
    dv = np.arctan2(np.sin(a[1]-b[1]), np.cos(a[1]-b[1]))
    return np.sqrt(du**2 + dv**2)

class Env:
    def __init__(self):
        self.rng = random.Random(42)
        self.reset()

    def reset(self):
        self.pos = np.array([
            [self.rng.random(), self.rng.uniform(0,2*np.pi)]
            for _ in range(NUM_SHARDS)
        ], dtype=np.float32)
        self.vel = np.zeros((NUM_SHARDS,2))
        self.leader = self.rng.randrange(NUM_SHARDS)
        return self.obs()

    def obs(self):
        o=[]
        for i in range(NUM_SHARDS):
            du = wrap(self.pos[self.leader][0], self.pos[i][0])
            dv = np.arctan2(
                np.sin(self.pos[self.leader][1]-self.pos[i][1]),
                np.cos(self.pos[self.leader][1]-self.pos[i][1])
            )
            o.append([self.pos[i][0], self.pos[i][1], du, dv])
        return np.array(o, dtype=np.float32)

    def step(self, act):
        self.vel = self.vel*0.85 + act
        self.pos += self.vel

        self.pos[:,0]%=1.0
        self.pos[:,1]%=2*np.pi

        handoff=False
        for i in range(NUM_SHARDS):
            if i!=self.leader and dist(self.pos[self.leader], self.pos[i])<0.2:
                self.leader=i
                handoff=True
                break

        avg_dist = np.mean([dist(self.pos[self.leader], self.pos[i]) for i in range(NUM_SHARDS)])
        energy = np.mean(np.linalg.norm(self.vel, axis=1))

        return self.obs(), -avg_dist + (20 if handoff else 0), energy

# ================= TRAIN =================
def train():
    env = Env()
    opt = optim.Adam(policy.parameters(), lr=3e-4)

    for ep in range(50):
        obs = env.reset()
        seq = [obs]*SEQ_LEN
        total = 0

        for _ in range(200):
            seq_t = torch.tensor(np.array(seq), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            pos_t = torch.tensor(env.pos, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            mean, val, hearts = policy(seq_t, pos_t)

            distn = torch.distributions.Normal(mean, torch.exp(policy.log_std))
            act = distn.sample()

            # ================= TYPED VETO =================
            act = act.clone()

            # soft dampening
            act *= (1 - hearts["black_soft"])

            # hard veto
            act *= (hearts["black_hard"] < 0.7).float()

            # federation voting (critical)
            if federation_vote(hearts["black_crit"]):
                act *= 0.0

            act_np = act[0].detach().cpu().numpy()

            obs, base_reward, energy = env.step(act_np)

            reward = base_reward * (1 + hearts["red"].mean().item())
            reward -= hearts["green"].mean().item() * energy * 0.05

            total += reward
            seq.append(obs)

        print(f"Ep {ep} | Reward {total:.2f}")

# ================= MAIN =================
if __name__ == "__main__":
    train()
