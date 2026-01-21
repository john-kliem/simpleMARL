# Core Imports
import os
import time 
from dataclasses import dataclass, field
import tyro 
from torch.utils.tensorboard import SummaryWriter

#Buffer/ML algorithms
import numpy as np 
import torch 
import random 
from simplemarl.vecenv import SerialVecEnv, ParallelVecEnv
from simplemarl.algorithms import ppo
from simplemarl.buffer import Buffer
#Environment Imports
from maritime_env import MaritimeRaceEnv


@dataclass
class Args:
    exp_name:str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 10
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""    
    env_id: str = "CPPyquaticus"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    num_envs: int = 100
    """the number of parallel game environments"""
    num_steps: int = 600
    """the number of steps to run in each environment per policy rollout"""
    minibatch_size: int = 120
    """the number of mini-batches"""
    update_epochs: int = 6
    """the K epochs to update the policy"""
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_minibatches: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    to_train: list = field(default_factory=lambda: ['agent_0'])
    """the ID's of agents to which will be trained"""
    policies:dict = field(default_factory=lambda:{'agent_0':"init_ppo"}) #Must contain policy for every agent in pettingzooenv
    device="cpu"#torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
def make_env():
    def thunk():
        env =  MaritimeRaceEnv()
        return env
    return thunk
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_minibatches = int(args.batch_size // args.minibatch_size)
    args.num_iterations = args.total_timesteps // args.batch_size
    avg = {}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    #obs_space, act_space, num_envs, num_steps):
    env = make_env()()
    obs_spaces = env.observation_spaces
    act_spaces = env.action_spaces
    buffers = {}
    policies = {}
    for aid in args.policies:
        if args.policies[aid] == "init_ppo":
            config = ppo.PPOConfig()
            config.num_iterations = args.num_iterations 
            config.device = args.device
            policies[aid] = ppo.PPO(obs_spaces[aid], act_spaces[aid], config)
        if aid in args.to_train:
            buffers[aid] = Buffer(obs_spaces[aid], act_spaces[aid], args.num_envs, args.num_steps)
        #TODO add loading PPO and DQN algorithms
    envs = SerialVecEnv(make_env, args.num_envs)
    for iteration in range(1, args.num_iterations+1):
        start_time = time.time()
        for aid in args.to_train:
            policies[aid].anneal_lr(iteration)
        #Anneal Learning Rate
        rets = envs.reset()
        for aid in args.to_train:
            rets[aid]['terms'] = np.ones((args.num_envs,),dtype=np.float32)
            rets[aid]['truncs'] = np.ones((args.num_envs,),dtype=np.float32)
        for i in range(args.num_steps):
            #Add Observation
            for aid in args.to_train:
                buffers[aid].observations[buffers[aid].get_step()] = torch.from_numpy(rets[aid]["obs"])
                buffers[aid].dones[buffers[aid].get_step()] = torch.from_numpy(np.logical_or(rets[aid]["terms"], rets[aid]["truncs"]).astype(np.float32))
            #Compute actions
            actions = {}
            for aid in policies:
                with torch.no_grad():
                    act, logprob, _, value = policies[aid].get_action_and_value(buffers[aid].observations[buffers[aid].get_step()])
                if aid in buffers:
                    buffers[aid].actions[buffers[aid].get_step()] = act.squeeze(-1) #{'actions':act.squeeze(-1), 'logprobs':logprob.squeeze(-1), 'values':value.squeeze(-1)})
                    buffers[aid].logprobs[buffers[aid].get_step()] = logprob.squeeze(-1)
                    buffers[aid].values[buffers[aid].get_step()] = value.squeeze(-1)
                actions[aid] = act.detach().cpu().numpy()
            envs.step_async(actions)
            rets = envs.step_wait() #obs, rew, term, trunc, info
            
            for aid in buffers:
                buffers[aid].rewards[buffers[aid].get_step()] = torch.from_numpy(rets[aid]['rews'])
                buffers[aid].step()
        #Bootstrap values in all buffers GAE
        for aid in buffers:
            buffers[aid].next_value = policies[aid].get_value(torch.from_numpy(rets[aid]['obs'])).squeeze(-1)
            buffers[aid].next_done = torch.from_numpy(np.logical_or(rets[aid]["terms"], rets[aid]["truncs"]).astype(np.float32))
            buffers[aid].calculate_returns_and_advantages(policies[aid].config.gamma, policies[aid].config.gae_lambda)

        #Update policy
        flat_batches = {}
        for aid in buffers:
            flat_batches[aid] = buffers[aid].get_flat_batch()
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size 
                mb_inds = b_inds[start:end]
                for aid in buffers:
                    minibatch = {k: v[mb_inds] for k, v in flat_batches[aid].items()}
                    policies[aid].update(minibatch)
        for aid in buffers:
            avg[aid] = buffers[aid].get_average_return()
            buffers[aid].reset()
        
        print(f"Iteration {iteration}/{args.num_iterations} | {(time.time()-start_time)} | Average: {avg}")
        
        #Log values to tensorboard
        #Display subset of values to stdout
