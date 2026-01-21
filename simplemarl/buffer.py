import numpy as np
import torch
from gymnasium.spaces import Box

class Buffer:
    def __init__(self, obs_space, act_space, num_envs, num_steps):
        #Variables required for setting buffer sizes
        self.num_envs = num_envs
        self.num_steps = num_steps 
        self.step = 0
        self.obs_space = obs_space 
        self.act_space = act_space 

        #Initialize Buffer
        self.observations = torch.zeros((num_steps, num_envs, *obs_space.shape), dtype=torch.float32)
        self.actions = torch.zeros((num_steps, num_envs, *act_space.shape), dtype=torch.float32)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    def add(self, data:dict):
        #TODO: Add safety check ensure dimensions are the same right now assumes each add is of size (num_envs)
        #TODO: Add check to make sure each addition contains obs, acts, rews, dones, actions, and values
        for k, v in data.items:
            if hasattr(self, k):
                #TODO Maybe don't copy pass in reference
                getattr(self,k)[self.step].detach().cpu().copy_(v)
        self.step += 1
    def reset(self):
        #Start overwriting the old values
        self.step = 0
    def get_values(self):
        return self.values[:self.step].reshape(-1)
    def get_returns(self):
        return self.returns[:self.step].reshape(-1)
    def get_rewards(self):
        return self.rewards[:self.step]
    def calculate_returns_and_advantages(self, next_value, next_done, gamma, gae_lambda):
        with torch.no_grad():
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps-1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t+1]
                    nextvalues = self.values[t+1]
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam 
            self.returns = self.advantages + self.values
                


