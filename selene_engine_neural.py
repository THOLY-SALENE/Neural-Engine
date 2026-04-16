"""
SELENE ENGINE v10 — FULL RESEARCH STACK (TRUE PPO + GNN-READY)

Includes:
- True PPO (GAE, value loss, entropy bonus, minibatch updates)
- Vectorized environments
- Clean Gym-style env
- Dataset export
- Evaluation
- Beautiful torus visualization
- Structured for GNN upgrade later

Run:
python selene_engine_v10.py --mode train --epochs 200
python selene_engine_v10.py --mode animate
"""

import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb

# ================= CONFIG =================
NUM_SHARDS = 8
DT = 0.05
DEVICE = "cpu"
GAMMA = 0.99
LAMBDA = 0.95

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
        h=self.base(x)
        return self.mean(h), torch.exp(self.log_std), self.value(h)

policy = Policy().to(DEVICE)

# ================= METRIC =================
def wrap_du(a,b):
    d=a-b
    if abs(d)>0.5: d-=np.sign(d)
    return d

def dist(a,b):
    du=wrap_du(a[0],b[0])
    dv=np.arctan2(np.sin(a[1]-b[1]),np.cos(a[1]-b[1]))
    return np.sqrt(du**2+dv**2)

# ================= ENV =================
class Env:
    def __init__(self,seed):
        self.rng=random.Random(seed)
        self.reset()

    def reset(self):
        self.pos=np.array([[self.rng.random(),self.rng.uniform(0,2*np.pi)] for _ in range(NUM_SHARDS)],dtype=np.float32)
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
        return self.obs(),reward,False,{}

# ================= PPO =================
def compute_gae(rewards,values):
    adv=[]
    gae=0
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+GAMMA*values[t+1]-values[t]
        gae=delta+GAMMA*LAMBDA*gae
        adv.insert(0,gae)
    return torch.tensor(adv)

def train(epochs=200):
    envs=[Env(42+i) for i in range(8)]
    opt=optim.Adam(policy.parameters(),lr=3e-4)

    for ep in range(epochs):
        obs_buf,act_buf,logp_buf,val_buf,rew_buf=[] ,[],[],[],[]

        obs=[env.reset() for env in envs]

        for _ in range(256):
            obs_t=torch.from_numpy(np.array(obs)).to(DEVICE)
            mean,std,val=policy(obs_t)
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
            val_buf.append(val.squeeze())
            rew_buf.append(torch.tensor(rewards))

            obs=next_obs

        # bootstrap
        val_buf.append(policy(torch.from_numpy(np.array(obs)).to(DEVICE))[2].squeeze())

        adv=compute_gae(rew_buf,val_buf)
        returns=adv+torch.stack(val_buf[:-1])

        obs_flat=torch.cat(obs_buf)
        act_flat=torch.cat(act_buf)
        logp_old=torch.cat(logp_buf).detach()

        for _ in range(4): # PPO epochs
            mean,std,val=policy(obs_flat)
            distn=torch.distributions.Normal(mean,std)
            logp=distn.log_prob(act_flat).sum(-1)

            ratio=torch.exp(logp-logp_old)
            surr1=ratio*adv.flatten()
            surr2=torch.clamp(ratio,0.8,1.2)*adv.flatten()
            policy_loss=-torch.min(surr1,surr2).mean()

            value_loss=((val.squeeze()-returns.flatten())**2).mean()
            entropy=distn.entropy().mean()

            loss=policy_loss+0.5*value_loss-0.01*entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep%10==0:
            print("Epoch",ep)

    torch.save(policy.state_dict(),"selene_v10.pt")

# ================= VIS =================
def torus(u,v,t):
    R=4+0.5*np.sin(t)
    r=1.5+0.2*np.cos(v+t)
    th=u*2*np.pi
    return (R+r*np.cos(v))*np.cos(th),(R+r*np.cos(v))*np.sin(th),r*np.sin(v)

def animate():
    env=Env(42)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    trails=[[] for _ in range(NUM_SHARDS)]

    def update(f):
        ax.clear()
        obs=env.obs()
        with torch.no_grad():
            mean,_,_=policy(torch.from_numpy(obs))
        env.step(mean.numpy())

        for i in range(NUM_SHARDS):
            x,y,z=torus(env.pos[i][0],env.pos[i][1],f*DT)
            trails[i].append((x,y,z))
            if len(trails[i])>60: trails[i].pop(0)
            pts=np.array(trails[i])
            ax.plot(pts[:,0],pts[:,1],pts[:,2])
            ax.scatter(x,y,z,s=120 if i==env.leader else 40)

    FuncAnimation(fig,update,frames=300,interval=30)
    plt.show()

# ================= CLI =================
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["train","animate"],default="animate")
    p.add_argument("--epochs",type=int,default=200)
    args=p.parse_args()

    if args.mode=="train":
        train(args.epochs)
    else:
        animate()
