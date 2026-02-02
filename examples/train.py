# Core Imports
import sys
import os
import time 
from dataclasses import dataclass, field
import tyro 
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np 
import torch 
import random 

#Buffer/ML algorithms
from simplemarl.vecenv import SerialVecEnv, ParallelVecEnv, SubProcVecEnv
from simplemarl.algorithms import ppo
from simplemarl.buffer import Buffer
from simplemarl.parallel_pet_wrapper import GymnasiumToPettingZooParallel

#Pyquaticus Environment Imports
from pyquaticus import pyquaticus_v0
from pyquaticus.mctf26_config import config_dict_std as mctf_config
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv



def run_game(env_fn, policies, num_games):
    env = env_fn()()
    rews = {aid:0.0 for aid in policies}
    for i in range(num_games):
        terms = {'agent_0':False}
        obs,_ = env.reset()
        for x in range(args.num_steps):
            if any(terms.values()):
                break
            actions = {}
            for aid in obs:
                with torch.no_grad():
                    acts,_,_,_ = policies[aid].get_action_and_value(torch.from_numpy(obs[aid]))
                actions[aid] = acts.detach().cpu().numpy()
            obs, r, terms, truncs,_ = env.step(actions)
            for aid in r:
                rews[aid] += r[aid]
    for aid in rews:
        rews[aid] /= num_games
    env.close()
    return rews
        


@dataclass
class Args:
    exp_name:str = os.path.basename(__file__)[:-len(".py")]
    save_path:str = "models"
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""    
    env_id: str = "CPPyquaticus"
    """the id of the environment"""
    total_timesteps: int = 15000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_workers: int = 40
    """Number of workers running num_envs environments"""
    num_steps: int = 600
    """the number of steps to run in each environment per policy rollout"""
    minibatch_size: int = 120
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_minibatches: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    to_train: list = field(default_factory=lambda: ['agent_0', 'agent_1', 
                                                    'agent_2', 'agent_3', 
                                                    'agent_4', 'agent_5'])
    """the ID's of agents to which will be trained"""
    policies:dict = field(default_factory=lambda:{'agent_0':"init_ppo", 
                                                  'agent_1':"init_ppo", 
                                                  'agent_2':"init_ppo",
                                                  'agent_3':"agent_2", 
                                                  'agent_4':'agent_1', 
                                                  'agent_5':'agent_0'}) #Must contain policy for every agent in pettingzooenv
    device:str="cpu"
def make_env():
    def thunk():
        import pyquaticus.utils.rewards as rew
        rews = {'agent_0':rew.caps_and_grabs,
                'agent_1':rew.caps_and_grabs,
                'agent_2':rew.caps_and_grabs,
                'agent_3':rew.caps_and_grabs,
                'agent_4':rew.caps_and_grabs,
                'agent_5':rew.caps_and_grabs}
        env = CompPyquaticusEnv(render_mode=None, config_dict=mctf_config, reward_config=rews)
        return env
    return thunk
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_workers*args.num_envs * args.num_steps)
    args.num_minibatches = int(args.batch_size // args.minibatch_size)
    args.num_iterations = args.total_timesteps // args.batch_size
    avg = {}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    global_step = 0
    env = make_env()()
    obs_spaces = env.observation_spaces
    act_spaces = env.action_spaces
    buffers = {}
    policies = {}
    logs = {}
    sw = {}

    # Initialize Model Paths if they don't already exist
    if not os.path.exists(os.path.dirname(__file__) + '/'+args.save_path):
        os.makedirs(os.path.dirname(__file__) + '/'+args.save_path)
    
    for aid in args.to_train:
        if not os.path.exists(os.path.dirname(__file__) + '/'+args.save_path + '/' + aid):
            os.makedirs(os.path.dirname(__file__) + '/'+args.save_path + '/' + aid)
    #TODO rework so agents not training don't require a full sized buffer
    for aid in args.policies:
        if args.policies[aid] == "init_ppo":
            config = ppo.PPOConfig()
            config.ent_coef = 0.1
            config.learning_rate = 2.5e-4
            config.num_iterations = args.num_iterations 
            config.device = args.device
            policies[aid] = ppo.PPO(obs_spaces[aid], act_spaces[aid], config)
        else:
            policies[aid] = policies[args.policies[aid]] # Use agents 3-5
        if aid in args.to_train:
            buffers[aid] = Buffer(obs_spaces[aid], act_spaces[aid], args.num_envs*args.num_workers, args.num_steps)
            sw[aid] = SummaryWriter(f"runs/{aid}")
            sw[aid].add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            global_step=0, 
            )
        avg[aid] = 0.0
        #TODO add loading PPO and DQN algorithms
    envs = SubProcVecEnv(make_env(), args.num_workers, args.num_envs)

    for iteration in range(1, args.num_iterations+1):
        start_time = time.time()
        for aid in args.to_train:
            policies[aid].anneal_lr(iteration)
        #Anneal Learning Rate
        rets = envs.reset()
        
        for i in range(args.num_steps):
            #Add Observation
            for aid in args.to_train:
                buffers[aid].observations[buffers[aid].get_step()].copy_(torch.from_numpy(rets[aid]["obs"]))
                buffers[aid].dones[buffers[aid].get_step()].copy_(torch.from_numpy(np.logical_or(rets[aid]["terms"], rets[aid]["truncs"]).astype(np.float32)))
                
            #Compute actions
            actions = {}
            for aid in policies:
                with torch.no_grad():
                    act, logprob, _, value = policies[aid].get_action_and_value(buffers[aid].observations[buffers[aid].get_step()])
                    if aid in buffers:
                        buffers[aid].actions[buffers[aid].get_step()].copy_(act.squeeze(-1)) #{'actions':act.squeeze(-1), 'logprobs':logprob.squeeze(-1), 'values':value.squeeze(-1)})
                        buffers[aid].logprobs[buffers[aid].get_step()].copy_(logprob.squeeze(-1))
                        buffers[aid].values[buffers[aid].get_step()].copy_(value.squeeze(-1))
                actions[aid] = act.detach().cpu().numpy()
            envs.step_async(actions)
            rets = envs.step_wait() #obs, rew, term, trunc, info
            
            for aid in buffers:
                buffers[aid].rewards[buffers[aid].get_step()].copy_(torch.from_numpy(rets[aid]['rews']))
                buffers[aid].step()
        #Bootstrap values in all buffers GAE
        for aid in buffers:
            buffers[aid].next_value = policies[aid].get_value(torch.from_numpy(rets[aid]['obs'])).squeeze(-1)
            buffers[aid].next_done = torch.from_numpy(np.logical_or(rets[aid]["terms"], rets[aid]["truncs"]).astype(np.float32))
            buffers[aid].calculate_returns_and_advantages(policies[aid].config.gamma, policies[aid].config.gae_lambda)

        #Update policy
        policy_update_start = time.time()
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
                    logs[aid] = policies[aid].update(minibatch)
        for aid in buffers:
            avg[aid] = buffers[aid].get_average_return()
            buffers[aid].reset()
        
        policy_update_elapsed = time.time() - policy_update_start
        global_step += args.num_envs * args.num_workers * args.num_steps
        print(f"Iteration {iteration}/{args.num_iterations} | Iteration Elapsed {(time.time()-start_time)} | Average: {avg} | SPS {((args.num_envs*args.num_workers*args.num_steps)/(time.time()-start_time))} | Policy Update {policy_update_elapsed}",flush=True)
        for a in args.to_train:
            if iteration % 10 == 0:
                torch.save(policies[a].state_dict(), f'./models/{a}/step_{global_step}') 
            sw[a].add_scalar("charts/episodic_return", avg[a], global_step)

            y_pred, y_true = buffers[a].get_values().detach().cpu().numpy(), buffers[a].get_returns().detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            sw[a].add_scalar("charts/learning_rate", policies[a].optimizer.param_groups[0]["lr"], global_step)
            sw[a].add_scalar("losses/value_loss", logs[a]["v_loss"], global_step)
            sw[a].add_scalar("losses/policy_loss", logs[a]["pg_loss"], global_step)
            sw[a].add_scalar("losses/entropy", logs[a]["entropy_loss"], global_step)
            sw[a].add_scalar("losses/old_approx_kl", logs[a]["old_approx_kl"], global_step)
            sw[a].add_scalar("losses/approx_kl", logs[a]["approx_kl"], global_step)
            sw[a].add_scalar("losses/clipfrac", logs[a]["clipfracs"], global_step)
            sw[a].add_scalar("losses/explained_variance", explained_var, global_step)
            sw[a].add_scalar("charts/SPS", int(((args.num_envs*args.num_workers*args.num_steps)/(time.time()-start_time))), global_step)
    envs.close()

    #Save Final Models
    for a in args.to_train:
        torch.save(policies[a].state_dict(), f'./models/{a}/step_{global_step}') 
        
