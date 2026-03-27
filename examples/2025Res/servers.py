import socket
import json
import subprocess
import sys
import argparse
import gymnasium as gym
import numpy as np
import pygame
# from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
import os
import pyquaticus.utils.rewards as rew
# from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
# from config import config_dict_std as required
import logging


from FeudalEffort.gen_config import config_dict_competition
from FeudalEffort.solution import solution as sol2
# from zdg.solution import solution as sol2
# from solution import solution as sol1
from models.solution import solution as sol1
# from solution_defender import solution as sol1

# from solution_defender import solution as sol1
data = {'captures':None, 'grabs':None, 'tags':None, 'collisions':None, 'agent_0':[], 'agent_1':[], 'agent_2':[], 'agent_3':[], 'agent_4':[], 'agent_5':[],
            'agent_0_oob':[], 'agent_0_has_flag':[],'agent_0_is_tagged':[],
            'agent_1_oob':[], 'agent_1_has_flag':[], 'agent_1_is_tagged':[],
            'agent_2_oob':[], 'agent_2_has_flag':[], 'agent_2_is_tagged':[],
            'agent_3_oob':[], 'agent_3_has_flag':[], 'agent_3_is_tagged':[],
            'agent_4_oob':[], 'agent_4_has_flag':[], 'agent_4_is_tagged':[],
            'agent_5_oob':[], 'agent_5_has_flag':[], 'agent_5_is_tagged':[]}
def unormalize(env, obs):
    unormalized = {}
    for k in obs:
        unormalized[k] = env.agent_obs_normalizer.unnormalized(obs[k])
    return unormalized

RENDER_MODE = None#'human'
if __name__ == "__main__":
   
    
    reward_config = {}
    config = config_dict_competition
    config['sim_speedup_factor'] = 4
    config['render_agent_ids'] = True
    env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict_competition,render_mode=RENDER_MODE, reward_config=reward_config, team_size=3)
    # state = {'agent_position':[[59.476,50.644],[55.360,25.85],[29.186,27.202],[109.502,32.136],[122.001,20.510],[126.765,36.078]], 
            #  'agent_heading':[ 292.2+90.0, 325.3+90.0,  17.0+90.0, 91.2+90.0, 92.3+90.0, 179.1+90.0,]} 
    
    obs, info = env.reset()#options={'init_dict':state})
    # import time 
    # time.sleep(20)
    # import sys
    # sys.exit()
    unormalized = unormalize(env, obs)
    
    cmd = ""
    sols1 = sol1(env)
    sols2 = sol2(env)
    while True:
        
        a0 = sols1.compute_action('agent_0',obs, unormalized, env._history_to_state(),info)
        a1 = sols1.compute_action('agent_1',obs, unormalized, env._history_to_state(),info)
        a2 = sols1.compute_action('agent_2',obs, unormalized, env._history_to_state(),info)
        a3 = sols2.compute_action('agent_3',obs, unormalized, env._history_to_state(),info)
        a4 = sols2.compute_action('agent_4',obs, unormalized, env._history_to_state(),info)
        a5 = sols2.compute_action('agent_5',obs, unormalized, env._history_to_state(),info)
        
        actions = {'agent_0':a0,'agent_1':a1,'agent_2':a2,'agent_3':a3, 'agent_4':a4,'agent_5':a5}
        
            
        #pos = self.state["agent_position"][agent.idx]
        data['agent_0'].append(env.state["agent_position"][0].tolist())
        data['agent_1'].append(env.state["agent_position"][1].tolist())
        data['agent_2'].append(env.state["agent_position"][2].tolist())
        data['agent_3'].append(env.state["agent_position"][3].tolist())
        data['agent_4'].append(env.state["agent_position"][4].tolist())
        data['agent_5'].append(env.state["agent_position"][5].tolist())
        for i in range(6):
            data[f"agent_{i}_oob"].append(env.state["agent_oob"][i].tolist())
            data[f"agent_{i}_has_flag"].append(env.state["agent_has_flag"][i].tolist())
            data[f"agent_{i}_is_tagged"].append(env.state["agent_is_tagged"][i].tolist())
        # print("Actions: ", actions)
        obs, rews, term, trunc, info = env.step(actions)
        unormalized = unormalize(env, obs)
        if term['agent_0'] or trunc['agent_0']:
            print("Game Ended")
            #DO NOT CHANGE ANYTHING BELOW THIS
            state = env.state 
            scores = ['captures', 'grabs', 'tags', 'agent_collisions']
            data['captures'] = state['captures'].tolist()
            data['grabs'] = state['grabs'].tolist()
            data['tags'] = state['tags'].tolist()
            data['collisions'] = state['agent_collisions'].tolist()
            with open('./hardware_recreate.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)
            for s in scores:
                print(s,":",state[s])
            print(f"Ratio: {sols2.get_ratio()}")
            print(f"Controller: {sols2.get_controller()}")
            sys.exit(0)