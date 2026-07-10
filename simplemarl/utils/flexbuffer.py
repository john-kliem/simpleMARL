import torch
from collections import defaultdict
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Union

class FlexBuffer:
    def __init__(self, fields, device):
        self.cstep = 0
        self.device = device
        self.timesteps = -1
        for name in fields.keys():
            shape, dtype = fields[name]
            setattr(self, name, torch.zeros(shape, dtype=dtype).to(self.device))
            if self.timesteps == -1:
                self.timesteps = shape[0]
        return
    def add(self, name:str, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        if not hasattr(self, name):
            return
        
        val = getattr(self,name)
        val[self.cstep] = x.to(self.device)
        return
    def reset(self):
        self.cstep = 0 
        for attribute in vars(self):
            val = getattr(self, attribute)
            if isinstance(val, torch.Tensor):
                val.zero_()
    def get(self, name:str) -> Union[torch.Tensor, None]:
        if not hasattr(self, name):
            return None
        return getattr(self, name)[self.cstep]
    def flatten(self, name:str) -> Union[torch.Tensor, None]:
        if not hasattr(self, name):
            return
        data = getattr(self, name)[:self.cstep]
        if(len(data.shape) >= 3 ):
            flat = data.reshape(-1, *data.shape[2:])
        else:
            flat = data.reshape(-1)
        return flat
    def to_device(self, device):
        self.device = device
        for attribute in vars(self):
            val = getattr(self, attribute)
            if isinstance(val, torch.Tensor):
                setattr(self, attribute, val.to(self.device))
        return
    def step(self,):
        self.cstep += 1
    def get_average_reward(self,):
        rewards = getattr(self, "rewards").sum(dim=0)
        dones = 1 + getattr(self, "dones").sum(dim=0)
        return (rewards / dones).mean()
    def returns_and_advantages(self, 
                               next_value:torch.Tensor,
                               next_done:torch.Tensor,
                               return_attr:str = "returns", 
                               value_attr:str = "values", 
                               advantage_attr:str = "advantages", 
                               reward_attr:str = "rewards",
                               done_attr:str = "dones",
                               gamma:float=0.99,
                               gae_lambda:float=0.95
                              ):
        """
        Computes returns and advantages using Generalized Advantage Estimation (GAE).

        This method populates the buffers specified by `return_attr` and `advantage_attr`.

        Args:
            next_value (torch.Tensor): The value of the state after the last step in the buffer.
            next_done (torch.Tensor): The done flag for the state after the last step.
            return_attr (str): Name of the buffer to store returns.
            value_attr (str): Name of the buffer containing value predictions for each step.
            advantage_attr (str): Name of the buffer to store advantages.
            rewards_attr (str): Name of the buffer containing rewards for each step.
            dones_attr (str): Name of the buffer containing done flags for each step.
            gamma (float): The discount factor.
            gae_lambda (float): The lambda parameter for GAE.
        """
        advantages = getattr(self, advantage_attr)
        rewards = getattr(self, reward_attr)
        values = getattr(self, value_attr)
        returns = getattr(self, return_attr)
        dones = getattr(self, done_attr)

        with torch.no_grad():
            lastgaelam = 0
            for t in reversed(range(self.timesteps)):
                if t == self.timesteps-1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value 
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam 
            returns[:] = advantages + values
        return



class FlexBuilder:
    def __init__(self,):
        """Initialize FlexBuffer Builder"""
        self._fields = {}
        return 
    def add(self, name: str, shape:tuple, dtype:torch.dtype):
        """
        Adds a field to be stored in the buffer.

        Args:
            name (str): The name of the data field (e.g., 'observation').
            shape (tuple): The shape of the tensor for a single entry.
            dtype (torch.dtype): The data type of the tensor (e.g., torch.float32).
        
        Returns:
            None
        """
        if not name.isidentifier():
            raise ValueError(f"Field name '{name}' is not a valid Python identifier")
        self._fields[name] = (shape, dtype)
        return 
    def build(self, device):
        """Build and returns the FlexBuffer instance"""
        return FlexBuffer(self._fields, device)
    
#Pre Configured Builders

#Holds all information needed to train PPO (Actor + Critic) or alternatively just an Actor 
def build_ippo(env_fn, agent, timesteps, num_envs, device):
    """Builds one buffer for 'agent' that contains everything needed for training a PPO algorithm"""
    env = env_fn()
    env.reset()
    buffer = FlexBuilder() 
    buffer.add("observations", shape=(timesteps,num_envs, *env.observation_space(agent).shape), dtype=torch.float32)
    buffer.add("actions", shape=(timesteps,num_envs, *env.action_space(agent).shape), dtype=torch.float32)
    buffer.add("values", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("advantages", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("returns", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("rewards", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("logprobs", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("dones", shape=(timesteps, num_envs), dtype=torch.float32)
    return buffer.build(device)
#Includes all attributes necessary for global critic
def build_mat_critic(env_fn, agents,timesteps, num_envs, device):
    """Builds one buffer for 'agent' that contains everything needed for training a PPO algorithm"""
    env = env_fn()
    env.reset()
    buffer = FlexBuilder() 
    obs_dim = env.observation_space(agents[agents[0]]).shape[0]
    buffer.add("joint_state", shape=(timesteps,num_envs, len(agents), obs_dim), dtype=torch.float32)
    buffer.add("joint_value", shape=(timesteps,num_envs,len(agents)), dtype=torch.float32)
    buffer.add("advantages", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("returns", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("rewards", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("dones", shape=(timesteps, num_envs), dtype=torch.float32)
    return buffer.build(device)

#Includes all attributes necessary for global critic
def build_critic(env_fn, agents,timesteps, num_envs, device):
    """Builds one buffer for 'agent' that contains everything needed for training a PPO algorithm"""
    env = env_fn()
    env.reset()
    buffer = FlexBuilder() 
    obs_dim = env.observation_space(agents[agents[0]]).shape[0]
    buffer.add("joint_state", shape=(timesteps,num_envs, len(agents), obs_dim), dtype=torch.float32)
    buffer.add("joint_value", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("advantages", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("returns", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("rewards", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("dones", shape=(timesteps, num_envs), dtype=torch.float32)
    return buffer.build(device)

def build_mappo(env_fn, agents, timesteps, num_envs, device):
    env = env_fn()
    env.reset()

    env = env_fn()
    env.reset()
    buffer = FlexBuilder() 
    buffer.add("observations", shape=(timesteps,num_envs, *env.observation_space(agents[0]).shape), dtype=torch.float32)
    buffer.add("actions", shape=(timesteps,num_envs, *env.action_space(agents[0]).shape), dtype=torch.float32)
    buffer.add("values", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("advantages", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("returns", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("rewards", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("logprobs", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("dones", shape=(timesteps, num_envs), dtype=torch.float32)

    #Joint State assumes every agent has same observation shape
    obs_dim = env.observation_space(agents[0]).shape[0]
    joint_obs_dim = obs_dim * len(agents) + len(agents)  # flattened joint obs
    buffer.add("joint_state", shape=(timesteps, num_envs, joint_obs_dim), dtype=torch.float32)  # flat
    return buffer.build(device)