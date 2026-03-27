import torch
from collections import defaultdict
import numpy as np
from gymnasium.spaces import spaces

class FlexBuffer:
    def __init__(self, fields, device):
        self.cstep = 0
        self.device = device
        for name in fields.keys():
            shape, dtype = fields[name]
            setattr(self, name, torch.zeros(shape=shape, dtype=dtype).to(self.device))
        return
    def add(self, name:str, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        if not hasattr(self, name):
            return
        val = getattr(self,name)
        val[self.cstep] = x
        return
    def reset(self):
        self.cstep = 0 
        for attribute in vars(self):
            val = getattr(self, attribute)
            if isinstance(val, torch.Tensor):
                val.zero_()
        
    def flatten(self, name:str) -> torch.Tensor | None:
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
                val.to(self.device)
        return


class FlexBuilder:
    def __init__(self,):
        """Initialize FlexBuffer Builder"""
        self._fields = set()
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
        return FlexBuffer(self._fields)
    
#Pre Configured Builders

def build_ippo(env_fn, agent_id, timesteps, num_envs, device):
    env = env_fn()()
    buffer = FlexBuilder() 
    buffer.add("observations", shape=(timesteps,num_envs,env.observation_space(agent_id)), dtype=torch.float32)
    buffer.add("actions", shape=(timesteps,num_envs,env.action_space(agent_id)), dtype=torch.float32)
    buffer.add("values", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("advantages", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("returns", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("rewards", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("logprobs", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("dones", shape=(timesteps, num_envs), dtype=torch.float32)
    return buffer.build()

def build_mappo(env_fn, agents, timesteps, num_envs, device):
    env = env_fn()()
    buffer = FlexBuilder() 

    state = 0
    for aid in agents:
        #Agent Specific Buffers
        buffer.add(f"a{aid}_observations", shape=(timesteps,num_envs,env.observation_space(aid)), dtype=torch.float32)
        buffer.add(f"a{aid}_actions", shape=(timesteps,num_envs,env.action_space(aid)), dtype=torch.float32)
        buffer.add(f"a{aid}_logprobs", shape=(timesteps,num_envs), dtype=torch.float32)
        buffer.add(f"a{aid}_rewards", shape=(timesteps,num_envs), dtype=torch.float32)
        state += env.get_obserservation_spaces(aid)[0]
    #Info For Critic
    #TODO Add Join STate
    buffer.add("state", shape=(timesteps,num_envs, state), dtype=torch.float32)
    buffer.add("rewards", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("values", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("advantages", shape=(timesteps,num_envs), dtype=torch.float32)
    buffer.add("returns", shape=(timesteps,num_envs), dtype=torch.float32)
    return buffer.build()