import numpy as np
import os
import torch
from torch import nn
from torch.distributions import Categorical
import torch.functional as F
import warnings
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Discrete
import copy
# Need an added import for competition submission?
# Post an issue to the github and we will work to get it added into the system!

# NOTE: You are only allowed to change the gen_config OBS params specified
# Changing additional variables will result in disqualification of that entry

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
ACTION_MAP = [ [1.0, -0.25],  [1.0, -0.5],  [1.0, -1],[1.0, -2], [1.0, -3],
               [1.0, -4],  [1.0, -5],  [1.0, -6],[1.0, -7], [1.0, -8], [1.0, -9], [1.0, -10], [1.0, -20], [1.0, -60], [1.0, -100],
               [0.0, 0], [1.0, 0.25],  [1.0, 0.5],  [1.0, 1],[1.0, 2], [1.0, 3],
               [1.0, 4],  [1.0, 5],  [1.0, 6],[1.0, 7], [1.0, 8], [1.0, 9], [1.0, 10], [1.0, 20], [1.0, 60], [1.0, 100]
]
# Load in your trained model and return the corresponding agent action based on the information provided in step()
class solution:
    # Add Variables required for solution

    def __init__(self,env):

        # Load in policy or anything else you want to load/do here
        # NOTE: You can only load from files that are in the same directory as the solution.py or a subdirectory

        # Load in learned policies see examples below:
        # = Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        # self.policy_two = hybrid_agent(
        #     test=True, anum=0
        # )  # Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        # self.policy_three = hybrid_agent(
        #     test=True, anum=0
        # )  # Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        obs_space =   Box(-1.0, 1.0, (61,), dtype=np.float32)
        act_space = Discrete(31)
        self.agent_0 = PPO(obs_space, act_space)
        self.agent_0.load_state_dict(torch.load('./models/agent_0/step_15000000'))
        self.agent_1 = PPO(obs_space, act_space)
        self.agent_1.load_state_dict(torch.load('./models/agent_1/step_15000000'))
        self.agent_2 = PPO(obs_space, act_space)
        self.agent_2.load_state_dict(torch.load('./models/agent_2/step_15000000'))
    # Given an observation return a valid action agent_id is agent that needs an action, observation space is the current normalized observation space for the specific agent
    def compute_action(
        self,
        agent_id: str,
        full_obs_normalized: dict,
        full_obs: dict,
        global_state: dict,info
    ):
        act = 0
        if agent_id == "agent_0" or agent_id =="agent_3":
            act = self.agent_0.get_action_and_value(torch.from_numpy(full_obs_normalized[agent_id]))[0].detach().cpu().numpy().item()
        elif agent_id == "agent_1" or agent_id =="agent_4":
            act = self.agent_1.get_action_and_value(torch.from_numpy(full_obs_normalized[agent_id]))[0].detach().cpu().numpy().item()
        elif agent_id == "agent_2" or agent_id =="agent_5":
            act = self.agent_2.get_action_and_value(torch.from_numpy(full_obs_normalized[agent_id]))[0].detach().cpu().numpy().item()
        # WARNING: If using global state you must ensure your entry can run on both RED and BLUE sides
        # State includes actual coordinate positions which are not the same on each side
        final_act =  copy.deepcopy(ACTION_MAP[act])
        final_act[0] = final_act[0]*3
        return final_act
        

# END OF CODE SECTION
