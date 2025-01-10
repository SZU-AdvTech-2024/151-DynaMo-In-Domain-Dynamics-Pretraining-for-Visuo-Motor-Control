import torch

from torch.optim import AdamW
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import os
import logging
import numpy as np
import sys
import time
# sys.path.append("D:/wangyc/idea/AI4SCI/Weather_Forcasting/ClimaX/src/climax/global_forecast/")
from new.downscaling.arch_downscaling import ClimaX
from Data_generation.datamodule import GlobalForecastDataModule
# from Encoder.encoder import Encoder
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    lat_weighted_mse_pde_loss_gradient,
    pearson,
)
sys.path.append("/home/jjh/new")
sys.path.append("/home/jjh/new/latent_foundation/dynamo_ssl-main")


def load_velocity(names):
    vel = []
    for name in names:
        vel.append(np.load("/home/hunter/workspace/climate/vel" + '/' + "{}.npy".format(name)))

    # 将所有张量按第一维度拼接在一起
    return [torch.from_numpy(v) for v in vel]

frame = 16 #记得改
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

model_name = "our_modelv3 no_interpolation_原始变量_schedu在里面"
default_vars = ['land_sea_mask', 'orography', 'lattitude', '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential_50', 'geopotential_250', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'geopotential_1000', 'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'u_component_of_wind_1000', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'v_component_of_wind_1000', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000', 'relative_humidity_50', 'relative_humidity_250', 'relative_humidity_500', 'relative_humidity_600', 'relative_humidity_700', 'relative_humidity_850', 'relative_humidity_925', 'relative_humidity_1000', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'specific_humidity_1000']

# default_vars = ["land_sea_mask",
#         "orography",
#         "lattitude",
#         "2m_temperature",
#         "10m_u_component_of_wind",
#         "10m_v_component_of_wind",
#         "geopotential_500",
#         "geopotential_600",
#         "temperature_700",
#         "temperature_850",
#         "relative_humidity_500",
#         "relative_humidity_850",
#         "specific_humidity_500",
#         "specific_humidity_850"]

default_vars=['land_sea_mask', 'orography', 'lattitude', '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential_50', 'geopotential_250', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'geopotential_1000', 'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'u_component_of_wind_1000', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'v_component_of_wind_1000', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000', 'relative_humidity_50', 'relative_humidity_250', 'relative_humidity_500', 'relative_humidity_600', 'relative_humidity_700', 'relative_humidity_850', 'relative_humidity_925', 'relative_humidity_1000', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'specific_humidity_1000']

def get_gauss_kernel(low_lat, low_lon, lat, lon, sigma=1.0):
    low_lat = torch.tensor(low_lat)
    low_lon = torch.tensor(low_lon)
    lat = torch.tensor(lat)
    lon = torch.tensor(lon)
    
    # Create meshgrid for low and high resolution coordinates
    low_coords = torch.stack(torch.meshgrid(low_lat, low_lon), dim=-1).reshape(-1, 2)
    high_coords = torch.stack(torch.meshgrid(lat, lon), dim=-1).reshape(-1, 2)
    
    # Calculate squared distances
    dist = torch.cdist(low_coords, high_coords, p=2).pow(2)
    
    # Calculate Gaussian kernel
    kernel = torch.exp(-dist / (2 * sigma**2))
    np.save("coordinate_kernel.npy", kernel)
  
def get_gauss_kernel_with_weights(low_lat, low_lon, lat, lon, sigma=1.0, target_h=48, target_w=96):
    # Convert input arrays to PyTorch tensors
    low_lat = torch.tensor(low_lat, dtype=torch.float32)
    low_lon = torch.tensor(low_lon, dtype=torch.float32)

    lat = torch.tensor(lat, dtype=torch.float32)
    lon = torch.tensor(lon, dtype=torch.float32)
    
    # Calculate latitude weights for low resolution coordinates
    w_low_lat = np.cos(np.deg2rad(low_lat.numpy()))
    w_low_lat = w_low_lat / w_low_lat.mean()  # Normalize
    w_low_lat = torch.from_numpy(w_low_lat).to(dtype=low_lat.dtype, device=low_lat.device)
    
    # Calculate latitude weights for high resolution coordinates
    w_lat = np.cos(np.deg2rad(lat.numpy()))
    w_lat = w_lat / w_lat.mean()  # Normalize
    w_lat = torch.from_numpy(w_lat).to(dtype=lat.dtype, device=lat.device)
    
    # Apply latitude weights to low and high latitude tensors
    weighted_low_lat = low_lat * w_low_lat
    weighted_lat = lat * w_lat
    
    # Create meshgrid for low and high resolution coordinates
    low_lat_grid, low_lon_grid = torch.meshgrid(weighted_low_lat, low_lon, indexing='ij')
    lat_grid, lon_grid = torch.meshgrid(weighted_lat, lon, indexing='ij')
    
    # # Compute the indices for the target dimensions using linspace
    # indices_low_lat = torch.linspace(0, low_lat_grid.shape[0]-1, target_h).long()
    # indices_low_lon = torch.linspace(0, low_lon_grid.shape[1]-1, target_w).long()

    # # Downsample the low resolution grids using the indices
    # low_lat_grid = low_lat_grid[indices_low_lat][:, indices_low_lon]
    # low_lon_grid = low_lon_grid[indices_low_lat][:, indices_low_lon]
    
    # Reshape grids to create coordinate pairs
    low_coords = torch.stack([low_lat_grid.flatten(), low_lon_grid.flatten()], dim=1)
    high_coords = torch.stack([lat_grid.flatten(), lon_grid.flatten()], dim=1)
    
    # Calculate squared distances
    dist = torch.cdist(low_coords, high_coords, p=2).pow(2)
    
    # Calculate Gaussian kernel
    kernel = torch.exp(-dist / (2 * sigma**2))
    
    # Normalize the kernel
    normalization_factor = torch.sqrt(
        torch.sum(kernel, dim=1, keepdim=True) * torch.sum(kernel, dim=0, keepdim=True)
    )
    normalized_kernel = kernel / normalization_factor
    
    return normalized_kernel

def low_downsample(x: torch.Tensor, target_h: int = 48, target_w: int = 96) -> torch.Tensor:
 
    _, _, _,h, w = x.shape
    
    # Compute the indices for the target dimensions using linspace
    indices_h = torch.linspace(0, h-1, target_h).long()
    indices_w = torch.linspace(0, w-1, target_w).long()
    
    # Use meshgrid to create the indices
    grid_h, grid_w = torch.meshgrid(indices_h, indices_w, indexing='ij')
    
    return x[:, :, :,grid_h, grid_w]

def ensure_directory_exists(directory_path):
    # 检查指定的路径是否存在
    if not os.path.exists(directory_path):
        # 如果不存在，创建新的文件夹
        os.makedirs(directory_path)
        print(f"文件夹'{directory_path}'已创建。")
    else:
        print(f"文件夹'{directory_path}'已存在。")


def generate_coord(lat, lon):
   
    lat = torch.tensor(lat, dtype=torch.float32)
    lon = torch.tensor(lon, dtype=torch.float32)
    
    # Calculate latitude weights for high resolution coordinates
    w_lat = np.cos(np.deg2rad(lat.numpy()))
    w_lat = w_lat / w_lat.mean()  # Normalize
    w_lat = torch.from_numpy(w_lat).to(dtype=lat.dtype, device=lat.device)
    
    # Apply latitude weights to low and high latitude tensors
    weighted_lat = lat * w_lat
    
    # Create meshgrid for low and high resolution coordinate
    
    
    return weighted_lat, lon


'''Loader Data'''
batch_size = 128
accumulation_steps = 1
pred_range = 1 #T
compete_len = 28000000
device = "cuda:2"

ssss = r"/home/jjh/new/downscaling/{}/".format(model_name)
num_epochs = 500 
stride_ = 2
logging_folder = ssss + str(pred_range)
H, W = 32, 64

out_vars = [
    "2m_temperature",
"10m_u_component_of_wind",
"10m_v_component_of_wind",
"geopotential_500",
"temperature_850",
]
low_data = GlobalForecastDataModule(buffer_size=100,root_dir=r'/home/hunter/workspace/climate/mnt/',variables=default_vars,out_variables=out_vars,batch_size=batch_size,predict_range=pred_range)
high_data = GlobalForecastDataModule(buffer_size=100,root_dir=r'/home/hunter/workspace/climate/mnt/2.8125deg_npz/',variables=out_vars,out_variables=out_vars,batch_size=batch_size,predict_range=pred_range)
high_data.setup()
low_data.setup()
low_lat, low_lon = low_data.get_lat_lon()
lat, lon = high_data.get_lat_lon()
val_clim = high_data.val_clim
test_clim = high_data.test_clim
normalization = high_data.transforms
mean_norm, std_norm = normalization.mean, normalization.std
mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
denormalization = transforms.Normalize(mean_denorm, std_denorm)


high_train_dataloader = high_data.train_dataloader()
low_train_dataloader = low_data.train_dataloader()
low_test_dataloader = low_data.test_dataloader()
high_test_dataloader = high_data.test_dataloader()
low_val_dataloader = low_data.val_dataloader()
high_val_dataloader = high_data.val_dataloader()




def train(model: ClimaX, epoch, low_dataloader, high_dataloader, optimizer, scheduler, device, accumulation_steps=6):
    model.train()
    total_loss = 0
    cnt = 0
    start_time = time.time()  # 开始时间记录

    # 清空梯度
    optimizer.zero_grad()
    
    for batch_idx, (high_batch, low_batch) in enumerate(zip(high_dataloader, low_dataloader)):
        batch_start_time = time.time()  # 记录每个 batch 的开始时间
        cnt += 1

        y, _, _, out_variables, _ = high_batch
        x,_,_, variables, _ = low_batch
        x = x[:, 0, :]
        y = y[:, 0, :]
        x = x.to(device)
        y = y.to(device)
        # print("shape of y:",y.shape)
        lead_times = torch.zeros(1).to(device)  # 创建一个包含单个 0 的张量

        # 前向传播和计算损失
        loss_dict, _ = model.forward(x, y, lead_times, variables, out_variables, [lat_weighted_mse], lat=lat)
        loss_dict = loss_dict[0]
        loss = loss_dict["loss"] / accumulation_steps  # 除以累计步数，平摊损失

        # 反向传播
        loss.backward()

        # 每隔 accumulation_steps 更新一次参数
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()  # 更新参数
            scheduler.step()
            optimizer.zero_grad()  # 清空梯度

            batch_time = time.time() - batch_start_time  # 计算每个 batch 的时间
            print(f"Epoch [{epoch}], Batch [{batch_idx+1}], Train Loss: {loss_dict}, Batch Time: {batch_time:.2f} s")
            logging.info(f"Epoch [{epoch}], Batch [{batch_idx+1}], Train Loss: {loss_dict}, Batch Time: {batch_time:.2f} s")

        total_loss += loss.item() * accumulation_steps  # 累加未经平摊的损失值

    # 处理未被整除的批次
    if cnt % accumulation_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    epoch_time = time.time() - start_time  # 计算整个 epoch 的时间

    return total_loss, epoch_time

# 测试/验证循环
def evaluate(model:ClimaX, low_dataloader, high_dataloader, device):
    model.eval()
    total_loss = 0
    all_loss_dicts = []
    total_batches = 0
    with torch.no_grad():
        cnt = 0
        for batch_idx, (high_batch, low_batch) in enumerate(zip(high_dataloader, low_dataloader)):
            total_batches += 1
            cnt += 1
            y, _, _, out_variables, _ = high_batch
            x,_,_, variables, _ = low_batch
            x = x[:,0,:]
            y = y[:,0,:]
            x = x.to(device)
            y = y.to(device)
            lead_times = torch.zeros(1)  # 创建一个包含单个 0 的张量
            lead_times = lead_times.to(device)
            if pred_range < 24:
                log_postfix = f"{pred_range}_hours"
            else:
                days = int(pred_range / 24)
                log_postfix = f"{days}_days"

            loss_dict = model.evaluate(
                x,
                y,
                lead_times,
                variables,
                out_variables,
                transform=denormalization,  # 假设 denormalization 已定义
                metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
                lat=lat,  # 假设 lat 已定义
                clim=test_clim,  # 假设 test_clim 已定义
                log_postfix=log_postfix,
            )
            if batch_idx % 250 == 0: print(batch_idx)
            all_loss_dicts.append(loss_dict)

    # 初始化合并后的损失字典
    loss_dict_combined = {}
    for dd in all_loss_dicts:
        for d in dd:
            for k in d.keys():
                if k in loss_dict_combined:
                    loss_dict_combined[k] += d[k]
                else:
                    loss_dict_combined[k] = d[k]

    # 计算平均值
    loss_dict_avg = {k: v / total_batches for k, v in loss_dict_combined.items()}

    return loss_dict_avg

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer, scheduler):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_test_loss = checkpoint['best_test_loss']
        best_val_epoch = checkpoint['best_val_epoch']
        best_test_epoch = checkpoint['best_test_epoch']
        return epoch - 1, best_val_loss, best_test_loss, best_val_epoch, best_test_epoch
    else:
        return 0, float('inf'), float('inf'), 0, 0

def main():
    ensure_directory_exists(logging_folder)

    lr = 5e-4
    betas = (0.9, 0.95)
    pde_weight = 0.01
    fourier_weight = 1.0

    logging.basicConfig(filename=os.path.join(logging_folder, "{},lr={}_pred_range={}_pdeweight={}.log".format(model_name,lr, pred_range, pde_weight)), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    jj, kk = generate_coord(low_lat, low_lon)
    jjj, kkk = generate_coord(lat, lon)

    # weights_path = r"/home/hunter/workspace/climate/Operator_for_climate_downscaling/Encoder/no_humidity/log-ep10.ckpt"
    # state_dict = torch.load(weights_path, map_location=device)
    checkpoint = torch.load("/home/jjh/new/latent_foundation/dynamo_ssl-main/exp_local/2024.12.05/162953_train_your_dataset_dynamo/snapshot/epoch298.pt")
    encoder = checkpoint["encoder"]

   
    model = ClimaX(default_vars=default_vars, img_size=[H, W], patch_size=2, grid_size=[jj, kk], high_gird=[jjj, kkk], pde_weight=pde_weight, out_dim=len(default_vars), encoder=encoder, fourier_weight=fourier_weight)
    
    # model.encoder.load_state_dict(state_dict)

    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    
    ipe = (6*(800-2))/(batch_size * accumulation_steps)
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=1e-5)
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5*ipe, eta_min=1e-8, max_epochs=num_epochs * ipe,warmup_start_lr=1e-8)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1 *ipe, eta_min=1e-8, max_epochs=num_epochs*ipe, warmup_start_lr=1e-8)
    model = model.to(device)
    start_epoch = 0
    # # 尝试加载检查点
    # start_epoch, best_val_loss, best_test_loss, best_val_epoch, best_test_epoch = load_checkpoint(f"{logging_folder}/our_bicubic_epoch=150,no_chazhi_climax_FFeatrue_kenel_all.pth.tar", model, optimizer, scheduler)
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_val_epoch = 0
    best_test_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, epoch, low_train_dataloader, high_train_dataloader, optimizer,scheduler, device, accumulation_steps)
        
        val_loss = evaluate(model, low_val_dataloader, high_val_dataloader,device)
        test_loss = evaluate(model, low_test_dataloader, high_test_dataloader,device)

        # 保存最小验证集损失对应的模型
        if val_loss['w_rmse'] < best_val_loss:
            best_val_loss = val_loss['w_rmse']
            best_val_epoch = epoch
            torch.save(model.state_dict(), f"{logging_folder}/{model_name}.pth")

        # 保存最小测试集损失对应的模型
        # if test_loss['w_rmse'] < best_test_loss:
        #     best_test_loss = test_loss['w_rmse']
        #     best_test_epoch = epoch
        #     torch.save(model.state_dict(), f"{logging_folder}/our_bicubic_epoch=150,no_chazhi_climax_FFeatrue_kenel_all.pth")


        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_test_loss': best_test_loss,
                'best_val_epoch': best_val_epoch,
                'best_test_epoch': best_test_epoch
            }, f"{logging_folder}/{model_name}.pth.tar")

        print(f"Epoch {epoch+1}, Val Loss: {val_loss}, Test Loss: {test_loss}")
        logging.info(f"Epoch {epoch+1}, Val Loss: {val_loss}, Test Loss: {test_loss}")

if __name__ == '__main__':
# 模型训练
    main()
