import time
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from dataclasses import dataclass
import torch.optim as optim
import math 

from transformer_blocks import MultiHeadAttentionBlock, EncoderBlock, Encoder, Decoder, DecoderBlock, ProjectionLayer, InputEmbedding, PositionalEncoding, Normalization, FeedForward, ResidualConnection

@dataclass
class TransformerConfig:
    seed: int = 10
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    
    tgt_vocab_size:int = 0
    """Size of output action space"""
    src_seq:int = 0
    """MAX sequence length of input"""
    tgt_seq:int = 0
    """MAX sequence length of output"""
    d_model:int = 512
    """Size of internal state representation"""
    num_blocks:int=1
    """number of encoder/decoder blocks"""
    num_heads:int=8
    """number of heads used in multi-head attention block"""
    dropout:float=0.1 
    """Probability of randomly setting NN activations to zero during training (prevent overfitting)"""
    d_ff:int=2048
    """Dimension of the hidden layer inside the Feed-Forward Network block"""
    vocab_size:int = 30
    """Size of single agent observation set at runtime"""
    critic_residual_connection:bool = True
    """Size of single agent observation set at runtime"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer




class CriticTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
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
        
        # We use nn.ModuleList so PyTorch tracks all sublayer weights
        self.blocks = nn.ModuleList([
            CriticTransformerBlock(self.config.d_model, self.config.num_heads, self.config.dropout) 
            for _ in range(self.config.num_blocks)
        ])
        
        # Final layers to project the transformer output to a state-value scalar
        self.final_norm = Normalization(self.config.d_model)

        #Maybe Make Value head MLP rather than one linear layer
        self.value_head = layer_init(nn.Linear(self.config.d_model, 1), std=1.0)
        self.optimizer = None 

        self.init_optimizer
    def init_optimizer(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def anneal_lr(self, iteration):
        if self.config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
            lrnow = frac * self.config.learning_rate 
            self.optimizer.param_groups[0]["lr"] = lrnow 
    def forward(self, x):
        # Process the sequence through each Transformer block
        for block in self.blocks:
            x = block(x)
            
        # Apply final normalization
        x = self.final_norm(x)
        
        # Pool across the sequence dimension (e.g., mean pool over agents/tokens)
        x = x.mean(dim=1) 
        
        # Output a single scalar state-value estimate (Value Function V(s))
        return self.value_head(x)
    def update(self, minibatch):
        return
    

class TransformerCritic(nn.Module):
    def __init__(self, config = TransformerConfig()):
        super().__init__()
        self.config = config
        
        # We use nn.ModuleList so PyTorch tracks all sublayer weights
        self.blocks = nn.ModuleList([
            CriticTransformerBlock(self.config.d_model, self.config.num_heads, self.config.dropout) 
            for _ in range(self.config.num_blocks)
        ])
        
        # Final layers to project the transformer output to a state-value scalar
        self.final_norm = Normalization(self.config.d_model)

        #Make Value head MLP
        self.value_head = layer_init(nn.Linear(self.config.d_model, 1), std=1.0)

    def forward(self, x):
        # Process the sequence through each Transformer block
        for block in self.blocks:
            x = block(x)
            
        # Apply final normalization
        x = self.final_norm(x)
        
        # Pool across the sequence dimension (e.g., mean pool over agents/tokens)
        x = x.mean(dim=1) 
        
        # Output a single scalar state-value estimate (Value Function V(s))
        return self.value_head(x)
    
class Actor(nn.Module):
    def __init__(self, config = PPOConfig):
        return False
