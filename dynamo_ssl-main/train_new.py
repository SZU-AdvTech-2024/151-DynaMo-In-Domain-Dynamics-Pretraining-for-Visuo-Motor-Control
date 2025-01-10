import os
import tqdm
import utils
import hydra
import torch
import einops
import datasets
import wandb
import numpy as np
import torch.distributed
from pathlib import Path
from datetime import timedelta
from omegaconf import OmegaConf
# from accelerate import Accelerator
from collections import OrderedDict
from workspaces.base import Workspace
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils
# from accelerate.logging import get_logger
import logging
import sys
sys.path.append('..')
from Data_generation.data_gen import create_data

os.environ["WANDB_START_METHOD"] = "thread"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("TrainerLogger")


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.effective_batch_size = self.cfg.batch_size
        utils.set_seed_everywhere(cfg.seed)
        self.job_num, self.work_dir = utils.get_hydra_jobnum_workdir()
        # print(f"Debug: job_num={self.job_num}, work_dir={self.work_dir}")
        logger.info("Saving to {}".format(self.work_dir))
        os.chdir(self.work_dir)
        self.work_dir = Path(os.getcwd())  # get the absolute path
        # self._init_tracker(cfg)
        self.device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # 实例化数据集
        self.train_loader = create_data(128) #DataLoader(self.train_set, shuffle=True, **kwargs)
        self.test_loader = create_data(128)
        self.dataset = {
            "train_loader": self.train_loader,
            "test_loader": self.test_loader,  # 可根据需求调整
        }
        # self.dataset = hydra.utils.instantiate(cfg.env.dataset)
        # self.train_set, self.test_set = self._split_and_slice_dataset(self.dataset)
        # self._setup_loaders(batch_size=self.cfg.batch_size)

        # Create the model
        self.encoder = None
        self.projector = None
        self.ssl = None
        self._init_encoder()
        self._init_projector()
        self._init_ssl()

        self.workspace: Workspace = hydra.utils.instantiate(
            self.cfg.env.workspace,
            cfg=self.cfg,
            work_dir=self.work_dir,
            _recursive_=False,
        )
        self.workspace.set_dataset(self.dataset)

        self.log_components = OrderedDict()
        self.epoch = 0

    def _init_tracker(self, cfg):
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_cfg["effective_batch_size"] = self.effective_batch_size
        wandb_cfg["save_path"] = str(self.work_dir)
        # 初始化W&B跟踪器
        wandb.init(project=cfg.project, config=wandb_cfg, reinit=False, settings=wandb.Settings(start_method="thread"))
        logger.info("wandb run url: %s", wandb.run.url)

    def _init_encoder(self):
        if self.encoder is None:  # possibly already initialized from snapshot
            self.encoder = hydra.utils.instantiate(self.cfg.encoder)
            self.encoder = self.encoder.to(self.device)
            # print("encoder'device is",next(self.encoder.parameters()).device)
            if self.cfg.sync_bn:
                self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.encoder
                )
            self.encoder_optim = torch.optim.AdamW(
                params=self.encoder.parameters(),
                lr=self.cfg.ssl_lr,
                weight_decay=self.cfg.ssl_weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _init_projector(self):
        if self.projector is None:  # possibly already initialized from snapshot
            self.projector = hydra.utils.instantiate(
                self.cfg.projector, _recursive_=False
            )
            self.projector = self.projector.to(self.device)
            # print("projector'device is",next(self.projector.parameters()).device)
            self.projector_optim: torch.optim.Optimizer = (
                self.projector.configure_optimizers(
                    lr=self.cfg.ssl_lr,
                    weight_decay=self.cfg.ssl_weight_decay,
                    betas=tuple(self.cfg.betas),
                )
            )
    
    def _init_ssl(self):
        if self.ssl is None:
            self.ssl = hydra.utils.instantiate(
                self.cfg.ssl,
                encoder=self.encoder,
                projector=self.projector,
            )
            self.ssl = self.ssl.to(self.device)
            # print("ssl'device is",next(self.ssl.parameters()).idia)

    def _split_and_slice_dataset(self, dataset):
        kwargs = {
            "train_fraction": self.cfg.train_fraction,
            "random_seed": self.cfg.seed,
            "window_size": self.cfg.window_size,
            "future_conditional": (self.cfg.goal_conditional == "future"),
            "min_future_sep": self.cfg.min_future_sep,
            "future_seq_len": self.cfg.goal_seq_len,
            "num_extra_predicted_actions": self.cfg.num_extra_predicted_actions,
        }
        return datasets.core.get_train_val_sliced(dataset, **kwargs)

    def _setup_loaders(self, batch_size=None, pin_memory=True, num_workers=None):
        if num_workers is None:
            num_workers = self.cfg.num_workers
        kwargs = {
            "batch_size": batch_size or self.cfg.batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        self.train_loader = DataLoader(self.train_set, shuffle=True, **kwargs)
        self.test_loader = DataLoader(self.test_set, shuffle=False, **kwargs)
        # self.train_loader = create_data(1) #DataLoader(self.train_set, shuffle=True, **kwargs)
        # self.test_loader = create_data(1)


    def train(self):
        if self.cfg.use_lr_scheduling:
            lr = self.adjust_lr()
            self.log_append("metrics", 1, {"lr": lr})
        self.ssl.adjust_beta(self.epoch, self.cfg.num_epochs)
        pbar = tqdm.tqdm(
            self.train_loader,
            desc=f"Training epoch {self.epoch}",
            # disable=not self.accelerator.is_main_process,
            ncols=80,
        )
        for data in pbar:
            obs, _, _ = data
            obs=obs.to(self.device)
            # print("obs‘device:",obs.device)
            # with self.accelerator.autocast():
            (
                obs_enc,
                obs_proj,
                ssl_loss,
                ssl_loss_components,
            ) = self.ssl.forward(obs)
            self.log_append("ssl_train", len(obs), ssl_loss_components)
            ssl_loss.backward(ssl_loss)
            # self.accelerator.backward(ssl_loss, retain_graph=True)

            if self.cfg.clip_grad_norm:
                nn_utils.clip_grad_norm_(self.encoder.parameters(), self.cfg.clip_grad_norm)
                nn_utils.clip_grad_norm_(self.projector.parameters(), self.cfg.clip_grad_norm)
                nn_utils.clip_grad_norm_(self.ssl.parameters(), self.cfg.clip_grad_norm)

            self.encoder_optim.step()
            self.projector_optim.step()
            self.ssl.step()

            self.encoder_optim.zero_grad(set_to_none=True)
            self.projector_optim.zero_grad(set_to_none=True)

    def eval(self):
        with utils.inference.eval_mode(
            self.encoder,
            self.projector,
            no_grad=True,
        ):
            # eval on test set
            self.eval_loss = 0
            for data in self.test_loader:
                obs, _, _ = data
                obs=obs.to(self.device)
                (
                    obs_enc,
                    obs_proj,
                    ssl_loss,
                    ssl_loss_components,
                ) = self.ssl.forward(obs)
                self.log_append(
                    "ssl_eval",
                    len(obs),
                    ssl_loss_components,
                )

                 # 计算编码器输出的均值和标准差
                flat_obs_enc = obs_enc.view(-1, obs_enc.size(-1))  # 将输出展平为 (N*T*V, E)
                obs_enc_mean_std = flat_obs_enc.std(dim=0).mean()
                obs_enc_mean_norm = flat_obs_enc.norm(dim=-1).mean()
                self.log_append(
                    "metrics",
                    len(flat_obs_enc),
                    {
                        "obs_enc_mean_std": obs_enc_mean_std,
                        "obs_enc_mean_norm": obs_enc_mean_norm,
                    },
                )

                 # 计算投影输出的均值和标准差
                flat_obs_proj = obs_proj.view(-1, obs_proj.size(-1))  # 将输出展平为 (N*T*V, Z)
                obs_proj_mean_std = flat_obs_proj.std(dim=0).mean()
                obs_proj_mean_norm = flat_obs_proj.norm(dim=-1).mean()
                self.log_append(
                    "metrics",
                    len(flat_obs_proj),
                    {
                        "obs_proj_mean_std": obs_proj_mean_std,
                        "obs_proj_mean_norm": obs_proj_mean_norm,
                    },
                )

    def run(self):
        # snapshot = Path(self.work_dir) / "snapshot.pt"
        snapshot = Path("/home/jjh/new/latent_foundation/dynamo_ssl-main/exp_local/2024.12.17/090014_train_your_dataset_dynamo/snapshot.pt")
        if snapshot.exists():
           print(f"Resuming: {snapshot}")
           self.load_snapshot(snapshot)
           self.epoch = 280

        self.train_iterator = tqdm.trange(
            self.epoch,
            self.cfg.num_epochs,
            # disable=not self.accelerator.is_main_process,
            ncols=80,
        )
        self.train_iterator.set_description("Training")
        # Reset the log.
        self.log_components = OrderedDict()
        min_eval = 0.0008547631155737896
        for epoch in self.train_iterator:
            self.epoch = epoch
            self.train()
            self.eval()
            eval_loss=self.flush_log(step=self.epoch, iterator=self.train_iterator)
            if eval_loss < min_eval:
                self.save_snapshot()
                min_eval=eval_loss
        return float(self.eval_loss)

    def save_snapshot(self):
        self._keys_to_save = [
            "encoder",
            "projector",
            "encoder_optim",
            "projector_optim",
            "ssl",
            "epoch",
        ]

        payload = {k: self.__dict__[k] for k in self._keys_to_save}
        snapshot_path = self.work_dir / "snapshot.pt"
        # 确保目录存在
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("wb") as f:
            torch.save(payload, f)
        logger.info(f"Snapshot saved, epoch: {self.epoch}.")

    def load_snapshot(self,snapshot_path):
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        # not_in_payload = set(self._keys_to_save) - set(payload.keys())
        # if len(not_in_payload):
        #     logger.warning("Keys not found in snapshot: %s", not_in_payload)

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value),
            )

    def flush_log(self, step, iterator=None):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        if iterator is not None:
            iterator.set_postfix_str(postfix)
        logger.info(f"[{self.job_num}] Epoch {self.epoch}: {log_components}")
        self.log_components = OrderedDict()
        return log_components.get('ssl_eval/total_loss')

    def adjust_lr(self):
        # from https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L420
        """Decays the learning rate with half-cycle cosine after warmup"""
        # fmt: off
        if self.epoch < self.cfg.warmup_epochs:
            lr = self.cfg.ssl_lr * self.epoch / self.cfg.warmup_epochs
        else:
            lr = self.cfg.ssl_lr * 0.5 * (1.0 + np.cos(np.pi * (self.epoch - self.cfg.warmup_epochs) / (self.cfg.num_epochs - self.cfg.warmup_epochs)))
        # fmt: on
        optimizers = [self.encoder_optim, self.projector_optim]
        for optim in optimizers:
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        return lr


@hydra.main(version_base="1.2", config_path="configs", config_name="train")
def main(cfg):
    trainer = Trainer(cfg)
    eval_loss = trainer.run()
    # print('eval_loss',float(eval_loss))
    return eval_loss


if __name__ == "__main__":
    main()
