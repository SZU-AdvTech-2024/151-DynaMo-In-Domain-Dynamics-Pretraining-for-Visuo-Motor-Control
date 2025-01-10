from Data_generation.datamodule import GlobalForecastDataModule
import torch
from torch.utils.data import DataLoader


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    # out = torch.stack([batch[i][1] for i in range(len(batch))])
    # lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][1]
    out_variables = batch[0][2]
    return (
        inp,
        [v for v in variables],
        [v for v in out_variables],
        
    )




def create_data(batch_size, using_var=['land_sea_mask', 'orography', 'lattitude', '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential_50', 'geopotential_250', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'geopotential_1000', 'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'u_component_of_wind_1000', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'v_component_of_wind_1000', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000', 'relative_humidity_50', 'relative_humidity_250', 'relative_humidity_500', 'relative_humidity_600', 'relative_humidity_700', 'relative_humidity_850', 'relative_humidity_925', 'relative_humidity_1000', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'specific_humidity_1000']):
    data = GlobalForecastDataModule(buffer_size=100,root_dir=r'/home/hunter/workspace/climate/mnt',variables=using_var,out_variables=["2m_temperature",
          "10m_u_component_of_wind",
          "10m_v_component_of_wind", 
          "geopotential_500",
          "temperature_850",],batch_size=batch_size,predict_range=0,start=0)

    data.setup()
    train =  data.data_train
    del data
    train_dataloader = DataLoader(
            train,
            collate_fn=collate_fn,
            batch_size=batch_size,
            drop_last=False,
            num_workers=5,
            persistent_workers=False)

    return train_dataloader

# data = create_data(batch_size)