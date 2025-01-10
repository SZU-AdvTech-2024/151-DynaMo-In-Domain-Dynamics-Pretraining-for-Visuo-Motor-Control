
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
import sys
import torch.nn.functional as F
from kymatio import Scattering2D
import numpy as np

sys.path.append("/home/hunter/workspace/climate/climate_predict/")
class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
    """

    def __init__(
        self,
        default_vars,
        encoder,
        time_range=4,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        encoder_depth=8,
        fuse_decoder_depth=2,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        grid_size=(32, 64),high_gird=(64,128), pde_weight=0.0001, fourier_weight=1.0,latent_dim=1024, emb_dim=1024, dec_emb_dim=768, dec_num_heads=16, dec_depth=3, num_mlp_layers=1, out_dim=5, eps=1e5, layer_norm_eps=1e-5, embedding_type="latlon"
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.encoder = encoder
        # Feature Extracer
        # --------------------------------------------------------------------------
        self.g = nn.Sequential(
            Mlp(in_features=2, hidden_features=100, out_features=100),
            nn.LayerNorm(100),
            Mlp(in_features=100, hidden_features=100, out_features=100)
        )
        self.f = nn.Sequential(
            Mlp(in_features=512, out_features=500),
            nn.LayerNorm(500),
            Mlp(in_features=500, out_features=500)
        )
        self.grid_size = grid_size
        self.scattering = Scattering2D(J=1, L=8, max_order=2, shape=(32, 64))

        # Create low grid and latents
        n_x, n_y = self.grid_size[0], self.grid_size[1]
        xx, yy = torch.meshgrid(n_x, n_y, indexing="ij")
        self.grid = torch.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])

        #high
        n_x_high, n_y_high = high_gird[0], high_gird[1]
        xx_high, yy_high = torch.meshgrid(n_x_high, n_y_high, indexing="ij")
        self.grid_high = torch.hstack([xx_high.flatten()[:, None], yy_high.flatten()[:, None]])
    
       
        # --------------------------------------------------------------------------




#---------------------------------------------------------------------------------
# Feature_Extracter
    def continouswavetransf(self, sig):
        B, C, H, W = sig.shape
        device = sig.device
        sig = sig.view(B * C, 1, H, W)
        sig = sig.cpu().detach().numpy()
        sig = self.scattering(sig)
        sig = torch.tensor(sig, device=device).view(B, C, -1)
        return sig

    def Featrue_Extracter(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, V, H, W]` shape.
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.f(x) #[B, V, L]
        return x
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Lat_lon Kernel
    def coord_encoding(self, x, b, coords):
        #coords.shape [H, W, 2] 
        H,W =  coords.shape[0], coords.shape[1]
        coords = coords.reshape(-1,2).to(x.device)
        coords = self.g(coords) #[B, H*W, L]
        coords = coords.unsqueeze(0).expand(b, -1, -1) 
        return coords
        #coords: [B, H, W, L]
    
    def get_gauss_kernel_with_weights(self, low_coords,high_coords, sigma=1.0, target_h=48, target_w=96):
        # [B, h*w, 2] [B, H*W, 2]
        # Calculate squared distances
        low_coords = self.q(low_coords)
        high_coords = self.q(high_coords)
        dist = torch.cdist(low_coords, high_coords, p=2).pow(2)
        
        # Calculate Gaussian kernel
        kernel = torch.exp(-dist / (2 * sigma**2))
        
        # Normalize the kernel
        normalization_factor = torch.sqrt(
            torch.sum(kernel, dim=1, keepdim=True) * torch.sum(kernel, dim=0, keepdim=True)
        )
        normalized_kernel = kernel / normalization_factor
        
        return normalized_kernel
#---------------------------------------------------------------------------------
    

#---------------------------------------------------------------------------------
# Operator
    def Operator(self, x, coords):
        # coords: [B, H, W, D]
        # x : [B,L,D]
        B, H, W = coords.shape[0], self.img_size[0] * 2, self.img_size[1] *2
        coords.reshape(B,H*W, coords.shape[-1]) #[B*H*W, D]
        x = self.f(x)
        x = x.reshape(B, 5, 100)
        coords = torch.einsum('bhl,bvl->bhv', coords, x)  # [B, H*W, V]

        coords = coords.reshape(B, H, W, -1)
        return coords #[B, H, W, Vo]
       
#---------------------------------------------------------------------------------


    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        # x = x[:,0,:]

        embed_U = self.encoder(x)
        expectation_c = self.coord_encoding(x, x.shape[0], self.grid_high)  #[B , H*W , L]


        preds = self.Operator(embed_U, expectation_c).permute(0,3,1,2) 
        preds = preds + F.interpolate(x[:,torch.tensor([3, 4, 5, 8, 35]).type(torch.long).to(x.device),:], size=(64, 128), mode='bicubic', align_corners=False)
        # F.interpolate(x, size=(64, 128), mode='bicubic', align_corners=False) #[B, H*W, Vo]
        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables,lat) for m in metric]  

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
