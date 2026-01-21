import time
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from dataclasses import dataclass
import torch.optim as optim

@dataclass
class PPOConfig:
    seed: int = 10
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CPPyquaticus"
    """the id of the environment"""
    total_timesteps: int = 5000000#30000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 600 #600
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 200#50#25#@50 per minibatch 50#25#10#250#250 # 50 @ 25 per minibatch
    """the number of mini-batches"""
    update_epochs: int = 6#10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1#Maybe should be 0.1 default 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class PPO(nn.Module):
    def __init__(self, obs_space, act_space, config=PPOConfig()):
        super().__init__()
        self.config = config
        self.optimizer = None
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )
        self.init_optimizer()
    def init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def anneal_lr(self, iteration):
        if self.config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
            lrnow = frac * self.config.learning_rate 
            self.optimizer.param_groups[0]["lr"] = lrnow 

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    #TODO: Save Load Optimizer
    def save(self, path=f"ppo_{time.time()}.pt"):
        torch.save(self.state_dict(), path)
    def load(self, path=None):
        assert path != None, "Error: Missing Argument 'path'; Cannot load a model without a path"
        return self.load_state_dict(torch.load(path))
    def update(self, mini_batch):
        clipfracs = []
        #Place minibatch onto correct device 
        #TODO: Maybe should just place larger batch earlier
        for k in mini_batch:
            if k == 'actions':
                mini_batch[k] = mini_batch[k].long().to(self.config.device)
            else:
                mini_batch[k] = mini_batch[k].to(self.config.device)

        _, newlogprob, entropy, newvalue = self.get_action_and_value(mini_batch['obs'], mini_batch['actions'])
        logratio = newlogprob - mini_batch['logprobs']
        ratio = logratio.exp()


        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio-1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
        # mini_batch['advant÷ages'] = (mini_batch['advantages'] - mini_batch['advantages'].mean()) / (mini_batch['advantages'].std() + 1e-8)
        if self.config.norm_adv:
            mini_batch['advantages'] = (mini_batch['advantages'] - mini_batch['advantages'].mean()) / (mini_batch['advantages'].std() + 1e-8)
        # Policy Loss
        pg_loss1 = -mini_batch['advantages'] * ratio 
        pg_loss2 = -mini_batch['advantages'] * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value Loss
        newvalue = newvalue.view(-1)
        if self.config.clip_vloss:
            v_loss_unclipped = (newvalue - mini_batch['returns']) **2
            v_clipped = mini_batch['values'] + torch.clamp(
                newvalue - mini_batch['values'],
                -self.config.clip_coef,
                self.config.clip_coef
            )
            v_loss_clipped = (v_clipped - mini_batch['returns']) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mini_batch['returns'])**2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

        self.optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        logs = {"v_loss":v_loss.item(), "pg_loss":pg_loss.item(), "entropy_loss":entropy_loss.item(), "old_approx_kl":old_approx_kl.item(), "approx_kl":approx_kl.item(), "clipfracs":np.mean(clipfracs)}
        return logs