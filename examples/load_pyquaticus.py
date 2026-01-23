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
from pyquaticus import pyquaticus_v0
from pyquaticus.mctf26_config import config_dict_std as mctf_config

from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import argparse

def make_env():
    import pyquaticus.utils.rewards as rew
    rews = {'agent_0':rew.caps_and_grabs,
            'agent_1':rew.caps_and_grabs,
            'agent_2':rew.caps_and_grabs,
            'agent_3':rew.caps_and_grabs,
            'agent_4':rew.caps_and_grabs,
            'agent_5':rew.caps_and_grabs}
    env = CompPyquaticusEnv(render_mode='human', config_dict=mctf_config, reward_config=rews)
    return env

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
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 2v2 PyQuaticus environment')
    parser.add_argument('agent_0', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-0-policy')
    parser.add_argument('agent_1', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    parser.add_argument('agent_2', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    args = parser.parse_args()
    env = make_env()
    policies = {'agent_0':PPO(env.observation_space('agent_0'), env.action_space('agent_0')),
                'agent_1':PPO(env.observation_space('agent_1'), env.action_space('agent_1')),
                'agent_2':PPO(env.observation_space('agent_2'), env.action_space('agent_2')),
                }
    policies['agent_0'].load_state_dict(torch.load(args.agent_0))
    policies['agent_1'].load_state_dict(torch.load(args.agent_1))
    policies['agent_2'].load_state_dict(torch.load(args.agent_2))
    obs,_ = env.reset()
    terms = {'agent_0':False}
    rsum = {'agent_0':0.0, 'agent_1':0.0, 'agent_2':0.0, 'agent_3':0.0, 'agent_4':0.0, 'agent_5':0.0}
    while not any(terms.values()):
        actions = {}
        for aid in obs:
            with torch.no_grad():
                if aid == "agent_0" or aid =="agent_3":
                    actions[aid] = policies["agent_0"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                elif aid == "agent_1" or aid =="agent_4":
                    actions[aid] = policies["agent_1"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                elif aid == "agent_2" or aid =="agent_5":
                    actions[aid] = policies["agent_2"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
        obs, rews, terms, truncs, _ = env.step(actions)
        print(f"Rewards: {rews}")
        for aid in rsum:
            rsum[aid] += rews[aid]
    print(f"Finale sum: {rsum}")