import torch
import torch.nn as nn
import math



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    @staticmethod 
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        #(batch, h, seq, d_k) --> (batch, h, seq, seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,-1e9)
        attention_weights = torch.softmax(attention_scores,dim=-1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)

        return (attention_weights @ value), attention_weights

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k) # (batch, seq, d_model) --> (batch, seq, d_model)
        value = self.w_v(v) # (batch, seq, d_model) --> (batch, seq, d_model)

        #(batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq, d_model) --> (batch, seq, d_model)  
        return self.w_o(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, features:int, attention_block:MultiHeadAttentionBlock, feed_forward:FeedForward, dropout:float):
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward = feed_forward 
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers=layers 
        self.norm = Normalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, features:int, attention_block:MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward:FeedForward, dropout:float)->None:
        super().__init__()
        self.attention = attention_block 
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward 

        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.attention(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x,encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers 
        self.norm = Normalization(features)
    def forward(self,x,encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        #(batch, seq, d_model) --> (batch, seq, vocab_size)
        return self.proj(x)



class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq, dropout):
        super().__init__()
        self.d_model = d_model 

        #Maximum Length of a sequence
        self.seq = seq 
        self.dropout = nn.Dropout(dropout)

        #Matrix of shape (seq, d_{model})
        pe = torch.zeros(seq, d_model)

        #(seq, 1)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False) # (batch, seq, d_model)
        #Ensure we don't over rely on certain features in training
        return self.dropout(x)# (batch, seq, d_model)
    

#Normalization Layer
class Normalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        #x: (batch, seq, hidden_size)
        mean = x.mean(dim=-1, keepdim=True) #(batch, seq, 1)
        std=x.std(dim=-1,keepdim=True,unbiased=False) #(batch, seq, 1)
        return self.alpha * (x-mean)/(std+self.eps) + self.bias
    
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # w1 and b1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        #(batch, seq, d_model) --> (batch, seq, d_ff) --> (batch, seq, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))