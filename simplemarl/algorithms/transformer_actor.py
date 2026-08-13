import time
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from dataclasses import dataclass
import torch.optim as optim
import math 

from simplemarl.algorithms.transformer_blocks import MultiHeadAttentionBlock, EncoderBlock, Encoder, Decoder, DecoderBlock, ProjectionLayer, InputEmbedding, PositionalEncoding, Normalization, FeedForward, ResidualConnection
from simplemarl.algorithms.transformer import TransformerConfig

#Vars to set in config

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, device):
        super().__init__()
        self.device = device
        self.mha = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.mha_residual_connection = ResidualConnection(d_model, dropout)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model,d_model), std=1.0),
        )
        self.residual_mlp_connections = ResidualConnection(d_model, dropout) 
        self.to(device)
    def forward(self, x):
        
        # Set of agent observations x
        x = self.mha_residual_connection(x, lambda y: self.mha(y, y, y, None))
        return self.residual_mlp_connections(x, self.mlp)


# Actor
# Architecture:
#   N Critic Blocks
#       MultiHead Attation
#       Add + Norm
#   MLP
#   Action
class TransformerActor(nn.Module):
    def __init__(self, config = TransformerConfig(), act_space=None):
        super().__init__()
        self.config = config
        self.act_space = act_space
        self.proj_layer = nn.Sequential(layer_init(nn.Linear(self.config.vocab_size, self.config.d_model)))
        # We use nn.ModuleList so PyTorch tracks all sublayer weights
        self.blocks = nn.ModuleList([
            ActorTransformerBlock(self.config.d_model, self.config.num_heads, self.config.dropout, self.config.device) 
            for _ in range(self.config.num_blocks)
        ])
        
        # Final layers to project the transformer output to a state-value scalar
        self.final_norm = Normalization(self.config.d_model)

        #Maybe Make Value head MLP rather than one linear layer
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.act_space), std=0.01),
        )
       
        self.optimizer = None 
        self.to(self.config.device)

        self.init_optimizer()
    def init_optimizer(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def anneal_lr(self, iteration):
        if self.config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
            lrnow = frac * self.config.learning_rate 
            self.optimizer.param_groups[0]["lr"] = lrnow
    
    def get_action(self, x, action=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.proj_layer(x)
        # Process the sequence through each Transformer block
        for block in self.blocks:
            x = block(x)
        # Apply final normalization
        x = self.final_norm(x)
        x = x[:,0]#x.mean(dim=1)
        # Output a single scalar state-value estimate (Value Function V(s))
        x = self.action_head(x)
        probs = Categorical(logits=x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def update(self, mini_batch):
        clipfracs = []
        #Place minibatch onto correct device 
        #TODO: Maybe should just place larger batch earlier

        _, newlogprob, entropy = self.get_action(mini_batch['obs'], mini_batch['actions'])
        logratio = newlogprob - mini_batch['logprobs']
        ratio = logratio.exp()


        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio-1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
        advantages = mini_batch['advantages'].clone()
        if self.config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Policy Loss
        pg_loss1 = -advantages * ratio 
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        
        entropy_loss = entropy.mean()
        loss = pg_loss - self.config.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        logs = {"v_loss":None, "pg_loss":pg_loss.item(), "entropy_loss":entropy_loss.item(), "old_approx_kl":old_approx_kl.item(), "approx_kl":approx_kl.item(), "clipfracs":np.mean(clipfracs)}
        return logs


class TransformerActorMulti(nn.Module):
    def __init__(self, config = TransformerConfig(), act_space=None):
        super().__init__()
        self.config = config
        self.act_space = act_space
        self.proj_layer = nn.Sequential(layer_init(nn.Linear(self.config.vocab_size, self.config.d_model)))
        # We use nn.ModuleList so PyTorch tracks all sublayer weights
        self.blocks = nn.ModuleList([
            ActorTransformerBlock(self.config.d_model, self.config.num_heads, self.config.dropout, self.config.device) 
            for _ in range(self.config.num_blocks)
        ])
        
        # Final layers to project the transformer output to a state-value scalar
        self.final_norm = Normalization(self.config.d_model)

        #Maybe Make Value head MLP rather than one linear layer
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.act_space[0]), std=0.01),
        )
        self.tag_head = nn.Sequential(
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.act_space[1]), std=0.01),
        )
        self.optimizer = None 
        self.to(self.config.device)

        self.init_optimizer()
    def init_optimizer(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def anneal_lr(self, iteration):
        if self.config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
            lrnow = frac * self.config.learning_rate 
            self.optimizer.param_groups[0]["lr"] = lrnow
    
    def get_action(self, x, action=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.proj_layer(x)
        # Process the sequence through each Transformer block
        for block in self.blocks:
            x = block(x)
        # Apply final normalization
        x = self.final_norm(x)
        x = x[:,0]#x.mean(dim=1)
        # Output a single scalar state-value estimate (Value Function V(s))
        move_logits = self.action_head(x)
        tag_logits = self.tag_head(x)
        move_probs = Categorical(logits=move_logits)
        tag_probs = Categorical(logits=tag_logits)
        
        if action is None:
            move_action = move_probs.sample()
            tag_action = tag_probs.sample()
        else:
            move_action = action[..., 0].long()
            tag_action  = action[..., 1].long()
            
        combined_action = torch.stack([move_action, tag_action], dim=-1)
        move_logprob = move_probs.log_prob(move_action)
        tag_logprob  = tag_probs.log_prob(tag_action)
        joint_logprob = move_logprob + tag_logprob
        joint_entropy = move_probs.entropy() + tag_probs.entropy()
        return combined_action, joint_logprob, joint_entropy

    def update(self, mini_batch):
        clipfracs = []
        #Place minibatch onto correct device 
        #TODO: Maybe should just place larger batch earlier

        _, newlogprob, entropy = self.get_action(mini_batch['obs'], mini_batch['actions'])
        logratio = newlogprob - mini_batch['logprobs']
        ratio = logratio.exp()


        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio-1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
        advantages = mini_batch['advantages'].clone()
        if self.config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Policy Loss
        pg_loss1 = -advantages * ratio 
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        
        entropy_loss = entropy.mean()
        loss = pg_loss - self.config.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        logs = {"v_loss":None, "pg_loss":pg_loss.item(), "entropy_loss":entropy_loss.item(), "old_approx_kl":old_approx_kl.item(), "approx_kl":approx_kl.item(), "clipfracs":np.mean(clipfracs)}
        return logs
    