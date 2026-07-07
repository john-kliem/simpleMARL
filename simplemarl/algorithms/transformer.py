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

@dataclass
class TransformerConfig:
    seed: int = 10
    """seed of the experiment"""
    device:str = "cpu"
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    tgt_size:int = 0
    """Size or shape of output action space"""
    src_seq:int = 5
    """MAX sequence length of input"""
    tgt_seq:int = 1
    """MAX sequence length of output"""
    d_model:int = 128
    """Size of internal state representation"""
    num_blocks:int=1
    """number of encoder/decoder blocks"""
    num_heads:int=2
    """number of heads used in multi-head attention block"""
    dropout:float=0.0 
    """Probability of randomly setting NN activations to zero during training (prevent overfitting)"""
    d_ff:int=d_model
    """Dimension of the hidden layer inside the Feed-Forward Network block"""
    vocab_size:int = 0
    """Size of single agent observation set at runtime"""
    critic_residual_connection:bool = True
    """Size of single agent observation set at runtime"""
    learning_rate:float = 2.5e-4
    clip_vloss:bool = True
    clip_coef:float = 0.2
    vf_coef:float = 0.5
    anneal_lr:bool = True
    num_iterations:int = 0
    max_grad_norm:float = 0.5
    norm_adv: bool = True
    """Toggles advantages normalization"""
    ent_coef:float = 0.01
    device = "cpu"
#Vars to set in config

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CriticTransformerBlock(nn.Module):
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


#Centralized Critic
# Architecture:
#   N Critic Blocks
#       MultiHead Attation
#       Add + Norm
#   MLP
#   State Value
class TransformerCritic(nn.Module):
    def __init__(self, config = TransformerConfig()):
        super().__init__()
        self.config = config
        self.proj_layer = nn.Sequential(layer_init(nn.Linear(self.config.vocab_size, self.config.d_model)))
        # We use nn.ModuleList so PyTorch tracks all sublayer weights
        self.blocks = nn.ModuleList([
            CriticTransformerBlock(self.config.d_model, self.config.num_heads, self.config.dropout, self.config.device) 
            for _ in range(self.config.num_blocks)
        ])
        
        # Final layers to project the transformer output to a state-value scalar
        self.final_norm = Normalization(self.config.d_model)

        #Maybe Make Value head MLP rather than one linear layer
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, 1), std=1.0),
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
            
    def get_value(self, x):
        #x = x.to(self.config.device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.proj_layer(x)
        # Process the sequence through each Transformer block
        for block in self.blocks:
            x = block(x)
            
        # Apply final normalization
        x = self.final_norm(x)
        x = x.mean(dim=1)
        # Output a single scalar state-value estimate (Value Function V(s))
        return self.value_head(x).squeeze(-1)
    
    def update(self, mini_batch):
        #Critic Update
        #Place minibatch onto correct device 
        #TODO: Maybe should just place larger batch earlier
        for k in mini_batch:
            if k == 'actions':
                mini_batch[k] = mini_batch[k]#.to(self.config.device)
            else:
                mini_batch[k] = mini_batch[k]#.to(self.config.device)

        newvalue = self.get_value(mini_batch['obs'])

        # Value Loss
        # newvalue = newvalue.view(-1)
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
        loss = v_loss * self.config.vf_coef

        self.optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        logs = {"v_loss":v_loss.item(), "pg_loss":None, "entropy_loss":None, "old_approx_kl":None, "approx_kl":None, "clipfracs":None}
        return logs