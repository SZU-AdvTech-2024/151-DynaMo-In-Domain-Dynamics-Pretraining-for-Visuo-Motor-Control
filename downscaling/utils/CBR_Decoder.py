import torch
import torch.nn as nn
import copy
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def has_nan(tensor):
    return torch.isnan(tensor).any().item()

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # print(type(x))
        return x + self.dropout(sublayer(self.norm(x)))


class CBRDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(CBRDecoder, self).__init__()
        self.layers = clones(layer, N)


    def forward(self, x, out_x, top_k, kb, nxt):
        count = 1
        for layer in self.layers:
            x = layer(x, out_x, top_k, kb, nxt, count)
            count += 1
        
        return x

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.LeakyReLU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.act(self.w_1(x))
        x = self.w_2(self.dropout(x))
        return x 


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.LeakyReLU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.act(self.w_1(x))
        x = self.w_2(self.dropout(x))
        return x 

class PositionwiseFeedForward1(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward1, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.LeakyReLU()
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.act(self.w_1(x))
        x = self.w_2(self.dropout(x))
        return x 




class CBR_DecoderLayer(nn.Module): #4435
    def __init__(self, size=1024, d_model=1024, d_ff=2048, h=16, dropout=0.1) -> None:
        super(CBR_DecoderLayer,self).__init__()
        self.size = size
        self.d_model = d_model
        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff1 = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.norm = nn.LayerNorm(1024)
        self.ff_kb = PositionwiseFeedForward1(d_model * 2 , d_ff*2, d_model ,dropout)


    def fusion_512_1024(self,x, kb, nxt):#5008
        ans = torch.zeros_like(x)
        n = len(kb)
        for k in range(n):
            t = self.attn(x, kb[k], nxt[k], None)
        
            ans = ans + t
            
        return self.norm(ans / n)

    def fusion_2048(self, x, out_x,top_k, kb, nxt):
        nxt = self.ff_kb(nxt)
        x = torch.cat((x, nxt), dim=1)

        pos = torch.zeros_like(x)
        pos[:,512:,:] = torch.tensor(1)
        
        x = pos + x

        x = self.attn(x, x, x, None)
        return x[:, 0:512, :] 


    def forward(self, x, out_x, top_k, kb, nxt, count):
        x = self.sublayer[0](x, lambda x: self.fusion_2048(x, out_x, top_k, kb, nxt))
        x = self.sublayer[1](x, self.ff1)
        x = self.sublayer[2](x, lambda x: self.attn(x, x, x))
        x = self.sublayer[3](x, self.ff1)
        return x 

def make_decoder(
    N=6
):
    model = CBRDecoder(CBR_DecoderLayer(), N)
    # print(type(model))
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.s
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='leaky_relu')
    return model
