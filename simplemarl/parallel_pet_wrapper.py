import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Space
from typing import Dict, Any, Optional

class GymnasiumToPettingZooParallel(ParallelEnv):
    """
    A wrapper that converts a standard Gymnasium environment into a 
    PettingZoo ParallelEnv with a single agent named 'agent_0'.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "gym_to_pc_v0"}

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        
        # Define the single agent
        self.agents = ["agent_0"]
        self.possible_agents = ["agent_0"]
        
        # Map spaces to the agent
        self.observation_spaces = {"agent_0": self.env.observation_space}
        self.action_spaces = {"agent_0": self.env.action_space}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Return observations mapped to agent_0
        observations = {"agent_0": obs}
        infos = {"agent_0": info}
        
        self.agents = self.possible_agents[:]
        return observations, infos

    def step(self, actions: Dict[str, Any]):
        # Extract action for agent_0
        action = actions["agent_0"]
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Wrap outputs in dictionaries
        observations = {"agent_0": obs}
        rewards = {"agent_0": reward}
        
        terminations = {"agent_0": terminated}
        truncations = {"agent_0": truncated}
        infos = {"agent_0": info}
        
        # If the episode ends, clear the agents list
        if terminated or truncated:
            self.agents = []
            
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]