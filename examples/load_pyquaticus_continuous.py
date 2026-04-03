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

def make_env():
    import pyquaticus.utils.rewards as rew
    rews = {'agent_0':rew.caps_and_grabs,
            'agent_1':rew.caps_and_grabs,
            'agent_2':rew.caps_and_grabs,
            'agent_3':rew.caps_and_grabs,
            'agent_4':rew.caps_and_grabs,
            'agent_5':rew.caps_and_grabs}
    mc_config = mctf_config
    mc_config['render_saving'] = True
    env = CompPyquaticusEnv(render_mode='human', config_dict=mctf_config, reward_config=rews, action_space='continuous')
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 3v3 PyQuaticus environment')
    parser.add_argument('agent_0', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-0-policy')
    parser.add_argument('agent_1', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    parser.add_argument('agent_2', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    parser.add_argument('agent_3', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-0-policy')
    parser.add_argument('agent_4', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    parser.add_argument('agent_5', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    
    args = parser.parse_args()
    env = make_env()
    policies = {'agent_0':ppo.PPOContinuous(env.observation_space('agent_0'), env.action_space('agent_0')),
                'agent_1':ppo.PPOContinuous(env.observation_space('agent_1'), env.action_space('agent_1')),
                'agent_2':ppo.PPOContinuous(env.observation_space('agent_2'), env.action_space('agent_2')),
                'agent_3':ppo.PPOContinuous(env.observation_space('agent_3'), env.action_space('agent_3')),
                'agent_4':ppo.PPOContinuous(env.observation_space('agent_4'), env.action_space('agent_4')),
                'agent_5':ppo.PPOContinuous(env.observation_space('agent_5'), env.action_space('agent_5')),
                }
    policies['agent_0'].load_state_dict(torch.load(args.agent_0))
    policies['agent_1'].load_state_dict(torch.load(args.agent_1))
    policies['agent_2'].load_state_dict(torch.load(args.agent_2))
    policies['agent_3'].load_state_dict(torch.load(args.agent_3))
    policies['agent_4'].load_state_dict(torch.load(args.agent_4))
    policies['agent_5'].load_state_dict(torch.load(args.agent_5))
    obs,_ = env.reset()
    terms = {'agent_0':False}
    rsum = {'agent_0':0.0, 'agent_1':0.0, 'agent_2':0.0, 'agent_3':0.0, 'agent_4':0.0, 'agent_5':0.0}
    steps = 0
    # print("Agent_0 Obs: ", obs['agent_0'] == obs['agent_5'])
    # print("Agent_1 Obs: ", obs['agent_1'] == obs['agent_4'])
    # print("Agent_2 Obs: ", obs['agent_2'] == obs['agent_3'])
    # time.sleep(5)
    # sys.exit()
    while not any(terms.values()):
        actions = {}
        for aid in obs:
            with torch.no_grad():
                if aid == "agent_0" or aid =="agent_3":
                    actions[aid] = policies["agent_0"].get_action_and_value(torch.from_numpy(obs[aid]))[0][0].detach().cpu().numpy()
                elif aid == "agent_1" or aid =="agent_4":
                    actions[aid] = policies["agent_1"].get_action_and_value(torch.from_numpy(obs[aid]))[0][0].detach().cpu().numpy()
                elif aid == "agent_2" or aid =="agent_5":
                    actions[aid] = policies["agent_2"].get_action_and_value(torch.from_numpy(obs[aid]))[0][0].detach().cpu().numpy()
        # print("Actions: ", actions)
        obs, rews, terms, truncs, _ = env.step(actions)
        print(f"Rewards: {rews}")
        for aid in rsum:
            rsum[aid] += rews[aid]
        steps += 1
    print(f"Finale sum: {rsum} Steps: {steps}")
    #Save Video
    env.buffer_to_video(recording_compression=True)
