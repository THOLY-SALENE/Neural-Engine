"""
SELENE ENGINE v13.1 — STABLE PPO FIX (SHAPE + TRAINING FIXES)

Key fixes:
- Correct tensor shaping for multi-env PPO
- Proper flattening of (T, N) → (T*N)
- Stable GAE + returns
- Matching log_prob / action shapes
- More stable training behavior

Run:
python selene_engine_v13_1.py --mode train
"""

import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb

# ===== CONFIG =====
NUM_SHARDS = 8
NUM_ENVS = 8
ROLLOUT = 256
DEVICE = "cpu"
GAMMA = 0.99
LAMBDA = 0.95

# ===== POLICY =====
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
        h=self.base(x)
        return self.mean(h), torch.exp(self.log_std), self.value(h)

policy = Policy().to(DEVICE)

# ===== ENV =====
def wrap_du(a,b):
    d=a-b
    if abs(d)>0.5: d-=np.sign(d)
    return d

def dist(a,b):
    du=wrap_du(a[0],b[0])
    dv=np.arctan2(np.sin(a[1]-b[1]),np.cos(a[1]-b[1]))
    return np.sqrt(du**2+dv**2)

class Env:
    def __init__(self,seed):
        self.rng=random.Random(seed)
        self.reset()

    def reset(self):
        self.pos=np.array([[self.rng.random(), self.rng.uniform(0,2*np.pi)] for _ in range(NUM_SHARDS)],dtype=np.float32)
        self.vel=np.zeros((NUM_SHARDS,2),dtype=np.float32)
        self.leader=self.rng.randrange(NUM_SHARDS)
        return self.obs()

    def obs(self):
        o=[]
        for i in range(NUM_SHARDS):
            du=wrap_du(self.pos[self.leader][0],self.pos[i][0])
            dv=np.arctan2(np.sin(self.pos[self.leader][1]-self.pos[i][1]),
                          np.cos(self.pos[self.leader][1]-self.pos[i][1]))
            o.append([self.pos[i][0],self.pos[i][1],du,dv])
        return np.array(o,dtype=np.float32)

    def step(self,act):
        self.vel=self.vel*0.85+act
        self.pos+=self.vel
        self.pos[:,0]%=1.0
        self.pos[:,1]%=2*np.pi

        for i in range(NUM_SHARDS):
            if i!=self.leader and dist(self.pos[self.leader],self.pos[i])<0.2:
                self.leader=i
                break

        reward=-np.mean([dist(self.pos[self.leader],self.pos[i]) for i in range(NUM_SHARDS)])
        return self.obs(), reward, False, {}

# ===== TRAIN =====
def train(epochs=150):
    envs=[Env(42+i) for i in range(NUM_ENVS)]
    opt=optim.Adam(policy.parameters(), lr=3e-4)

    for ep in range(epochs):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []

        obs=[env.reset() for env in envs]

        for t in range(ROLLOUT):
            obs_t=torch.tensor(obs, dtype=torch.float32)
            mean,std,val = policy(obs_t)

            distn=torch.distributions.Normal(mean,std)
            act=distn.sample()
            logp=distn.log_prob(act).sum(-1)

            next_obs=[]
            rewards=[]
            for i,env in enumerate(envs):
                o,r,_,_=env.step(act[i].detach().numpy())
                next_obs.append(o)
                rewards.append(r)

            obs_buf.append(obs_t)
            act_buf.append(act)
            logp_buf.append(logp)
            val_buf.append(val.squeeze(-1))
            rew_buf.append(torch.tensor(rewards))

            obs=next_obs

        # bootstrap
        last_val = policy(torch.tensor(obs, dtype=torch.float32))[2].squeeze(-1)
        val_buf.append(last_val)

        # convert to tensors (T, N)
        val_buf = torch.stack(val_buf)
        rew_buf = torch.stack(rew_buf)

        # GAE
        adv = torch.zeros_like(rew_buf)
        gae = torch.zeros(NUM_ENVS)

        for t in reversed(range(ROLLOUT)):
            delta = rew_buf[t] + GAMMA * val_buf[t+1] - val_buf[t]
            gae = delta + GAMMA * LAMBDA * gae
            adv[t] = gae

        returns = adv + val_buf[:-1]

        # flatten (T*N, ...)
        obs_flat = torch.cat(obs_buf)
        act_flat = torch.cat(act_buf)
        logp_old = torch.cat(logp_buf).detach()
        adv_flat = adv.reshape(-1)
        returns_flat = returns.reshape(-1)

        # normalize adv
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for _ in range(4):
            mean,std,val = policy(obs_flat)
            distn=torch.distributions.Normal(mean,std)
            logp=distn.log_prob(act_flat).sum(-1)

            ratio=torch.exp(logp - logp_old)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio,0.8,1.2) * adv_flat

            policy_loss = -torch.min(surr1,surr2).mean()
            value_loss = 0.5 * ((val.squeeze(-1) - returns_flat)**2).mean()
            entropy = distn.entropy().mean()

            loss = policy_loss + value_loss - 0.01 * entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % 20 == 0:
            print(f"Epoch {ep}")

    torch.save(policy.state_dict(), "selene_v13_1.pt")

# ===== CLI =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
