import os
import time 
from dataclasses import dataclass, field
import tyro 
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
#Buffer/ML algorithms
import numpy as np 
import torch 
import random 
from simplemarl.vecenv import SerialVecEnv, ParallelVecEnv, SubProcVecEnv
from simplemarl.algorithms import ppo
from simplemarl.buffer import Buffer
from simplemarl.parallel_pet_wrapper import GymnasiumToPettingZooParallel
#Environment Imports
# from maritime_env import MaritimeRaceEnv
# from pyquaticus import pyquaticus_v0
# from pyquaticus.config import config_dict_std as mctf_config

# from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv

from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std as mctf_config
# from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from maritime_env.renderer import PygameRenderer

import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import argparse
from maritime_env.maritime_race_env import MaritimeRaceEnv

def make_env():
    def thunk():
        env = MaritimeRaceEnv(render_mode='human')
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 3v3 PyQuaticus environment')
    parser.add_argument('agent_0', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-0-policy')
   
    
    args = parser.parse_args()
    env = make_env()()
    print("Obs Space: ", env.observation_space('agent_0'), " Action Space: ", env.action_space('agent_0'))
    policies = {'agent_0':PPO(env.observation_space('agent_0'), env.action_space('agent_0')),}
    policies['agent_0'].load_state_dict(torch.load(args.agent_0))
    renderer = PygameRenderer()
    
        
    obs,_ = env.reset()
    terms = {'agent_0':False,}
    rsum = {'agent_0':0.0, }
    steps = 0
    while not any(terms.values()):
        actions = {}
        for aid in obs:
            with torch.no_grad():
                if aid == "agent_0":
                    actions[aid] = policies["agent_0"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy().item()
        
        
        # print("Actions: ", actions)
        obs, rews, terms, truncs, _ = env.step(actions)
        if rews['agent_0'] > 0:
            print(f"Rewards: {rews['agent_0']}")
        for aid in rsum:
            rsum[aid] += rews[aid]
        steps += 1
        renderer.render(env.c_env.get_boat_pos())
        time.sleep(0.25)
    print(f"Final sum: {rsum} Steps: {steps}")
    #Save Video if enabled
    
    env.close()
