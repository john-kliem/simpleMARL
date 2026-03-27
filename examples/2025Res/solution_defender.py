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
from pyquaticus.config import get_std_config, ACTION_MAP
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
# Need an added import for competition submission?
# Post an issue to the github and we will work to get it added into the system!

# NOTE: You are only allowed to change the gen_config OBS params specified
# Changing additional variables will result in disqualification of that entry

# Load in your trained model and return the corresponding agent action based on the information provided in step()
class solution:
    # Add Variables required for solution

    def __init__(self, env):

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
        self.env = env
        obs_space =   Box(-1.0, 1.0, (61,), dtype=np.float32)
        act_space0 = Discrete(31)
        self.agent_0 = BaseDefender(
            env.agents_of_team[Team.BLUE_TEAM][0].id,
            Team.BLUE_TEAM,
            self.env,
            mode='easy',
            continuous=True
        )
        self.agent_1 = BaseDefender(
            env.agents_of_team[Team.BLUE_TEAM][1].id,
            Team.BLUE_TEAM,
            self.env,
            mode='easy',
            continuous=True
        )
        self.agent_2 = BaseDefender(
            env.agents_of_team[Team.BLUE_TEAM][2].id,
            Team.BLUE_TEAM,
            self.env,
            mode='easy',
            continuous=True
        )

        self.agent_3 = BaseDefender(
            env.agents_of_team[Team.RED_TEAM][0].id,
            Team.RED_TEAM,
            self.env,
            mode='easy',
            continuous=True
        )
        self.agent_4 = BaseDefender(
            env.agents_of_team[Team.RED_TEAM][1].id,
            Team.RED_TEAM,
            self.env,
            mode='easy',
            continuous=True
        )
        self.agent_5 = BaseDefender(
            env.agents_of_team[Team.RED_TEAM][2].id,
            Team.RED_TEAM,
            self.env,
            mode='easy',
            continuous=True
        )
    # Given an observation return a valid action agent_id is agent that needs an action, observation space is the current normalized observation space for the specific agent
    def compute_action(
        self,
        agent_id: str,
        full_obs_normalized: dict,
        full_obs: dict,
        global_state: dict,
        info
    ):
        
        act = 0
        if agent_id == "agent_0":
            act = self.agent_0.compute_action(full_obs_normalized[agent_id], info)
        elif agent_id == "agent_1":
            act = self.agent_1.compute_action(full_obs_normalized[agent_id], info)
        elif agent_id == "agent_2":
            act = self.agent_2.compute_action(full_obs_normalized[agent_id], info)
        elif agent_id == "agent_3":
            act = self.agent_3.compute_action(full_obs_normalized[agent_id], info)
        elif agent_id == "agent_4":
            act = self.agent_4.compute_action(full_obs_normalized[agent_id], info)
        elif agent_id == "agent_5":
            act = self.agent_5.compute_action(full_obs_normalized[agent_id], info)
        # WARNING: If using global state you must ensure your entry can run on both RED and BLUE sides
        # State includes actual coordinate positions which are not the same on each side
        final_act =  [act[0]*3, act[1]]
        return final_act
        

# END OF CODE SECTION
