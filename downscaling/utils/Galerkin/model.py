from tkinter import NO

from workspace.climate.climate_predict.src.climax.DiS.models_dis import PatchEmbed
from .layers import *
import copy
import os
import sys
from collections import defaultdict
from typing import Optional
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
import torch
import torch.nn as nn
from torch import Tensor


current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
sys.path.append(SRC_ROOT)

ADDITIONAL_ATTR = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation', 
                   'downscaler_activation', 'upscaler_activation',
                   'encoder_dropout', 'decoder_dropout', 'ffn_dropout']
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GalerkinTransformerDecoderLayer(nn.Module):
    r"""
    A lite implementation of the decoder layer based on linear causal attention
    adapted from the TransformerDecoderLayer in PyTorch
    https://github.com/pytorch/pytorch/blob/afc1d1b3d6dad5f9f56b1a4cb335de109adb6018/torch/nn/modules/transformer.py#L359
    """
    def __init__(self, d_model=1024, 
                        nhead=16,
                        pos_dim = 1,
                        dim_feedforward=2048, 
                        attention_type='galerkin',
                        layer_norm=False,
                        attn_norm=None,
                        norm_type='layer',
                        norm_eps=1e-5,
                        xavier_init: float=1e-2,
                        diagonal_weight: float = 1e-2,
                        dropout=0.05, 
                        ffn_dropout=None,
                        activation_type='relu',
                        device=None, 
                        dtype=None,
                        debug=False,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype, }
        super(GalerkinTransformerDecoderLayer, self).__init__()

        ffn_dropout = default(ffn_dropout, dropout)
        self.debug = debug
        self.self_attn = SimpleAttention(nhead, d_model, 
                                        attention_type=attention_type,
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        self.multihead_attn = SimpleAttention(nhead, d_model, 
                                        attention_type='None',
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.add_layer_norm = layer_norm
        if self.add_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x: Tensor, kb: Tensor, nxt:Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.add_layer_norm:
            x = self.norm2(x + self._sa_block(x, None))
            x = self.norm3(x + self._ff_block(x))
        else:
            x = x + self._sa_block(x, tgt_mask)
            x = x + self._mha_block(x, kb, nxt, memory_mask)
            x = x + self._ff_block(x)
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, None)[0]
        return self.dropout2(x)

    def fusion(self,x, kb, nxt):#5008
        ans = torch.zeros_like(x)
        n = len(kb)
        for k in range(n):
            t = self.multihead_attn(x, kb[k], nxt[k], None)[0]
        
            ans = ans + t
            
        return ans / n


    # multihead attention block
    def _mha_block(self, x: Tensor, kb: Tensor, nxt:Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        # x = self.multihead_attn(x, mem, mem, mask=attn_mask,)[0]
        x = self.fusion(x, kb, nxt)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.ff(x)
        return self.dropout(x)

class GalerkinDeocder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(GalerkinDeocder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, kb, nxt):
        for layer in self.layers:
            x = layer(x, kb , nxt)

        return x


class PointwiseRegressor(nn.Module):
    def __init__(self, in_dim = 1024,  # input dimension
                 n_hidden = 1024,
                 out_dim = 5,  # number of target dim
                 num_layers: int = 1,
                 spacial_fc: bool = True,
                 spacial_dim=2,
                 dropout=0.1,
                 activation='relu',
                 return_latent=False,
                 debug=False):
        super(PointwiseRegressor, self).__init__()
        '''
        A wrapper for a simple pointwise linear layers
        '''
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        activ = nn.SiLU() if activation == 'silu' else nn.ReLU()
        if self.spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        self.ff = nn.ModuleList([nn.Sequential(
                                nn.Linear(n_hidden, n_hidden),
                                activ,
                                )])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activ,
            ))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.ff:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)

        if self.return_latent:
            return x, None
        else:
            return x

class GalerkinTransformer(nn.Module):
    r"""
    A lite implementation of the decoder layer based on linear causal attention
    adapted from the TransformerDecoderLayer in PyTorch
    https://github.com/pytorch/pytorch/blob/afc1d1b3d6dad5f9f56b1a4cb335de109adb6018/torch/nn/modules/transformer.py#L359
    """
    def __init__(self, d_model=1024, 
                        nhead=16,
                        pos_dim = 1,
                        dim_feedforward=2048, 
                        attention_type='galerkin',
                        layer_norm=False,
                        attn_norm=None,
                        norm_type='layer',
                        norm_eps=1e-5,
                        xavier_init: float=1e-2,
                        diagonal_weight: float = 1e-2,
                        dropout=0.05, 
                        ffn_dropout=None,
                        activation_type='relu',
                        device=None, 
                        dtype=None,
                        debug=False,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype, }
        super(GalerkinTransformerDecoderLayer, self).__init__()

        ffn_dropout = default(ffn_dropout, dropout)
        self.debug = debug
        self.self_attn = SimpleAttention(nhead, d_model, 
                                        attention_type=attention_type,
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        self.multihead_attn = SimpleAttention(nhead, d_model, 
                                        attention_type='causal',
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.add_layer_norm = layer_norm
        if self.add_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
     


    def forward(self, x: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.add_layer_norm:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))
        else:
            x = x + self._sa_block(self.norm1(x), None)
            x = x + self._ff_block(self.norm1(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask,)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem, mask=attn_mask,)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.ff(x)
        return self.dropout(x)


class ClimaX(nn.Module):
    def __init__(self, N=8, img_size=[32,64], patch_size=2, C=5, embed_dim=1024, output_dim=5):
        super(ClimaX, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, C, embed_dim)
        self.decoder = nn.ModuleList([copy.deepcopy(GalerkinTransformer()) for _ in range(N)])
        self.regressor = PointwiseRegressor(embed_dim, output_dim)
    
    def forward(self, x, coordinate):
        x = self.patch_embed(x)
        x = self.decoder(x)
        x = self.regressor(x, coordinate)
        return x




def make_GalerkinTransformerDeocder(N=8):
    
    model = GalerkinDeocder(GalerkinTransformerDecoderLayer(), N)
    
    return model


def make_GalerkinTransformer(N=8, img_size=[32,64], patch_size = 2, C=5, embed_dim=1024):
    
    model = GalerkinDeocder(GalerkinTransformer(), N)
    
    return model
