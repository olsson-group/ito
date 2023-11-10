import math

import pytorch_lightning as pl
import torch
from torch import nn
from torch_scatter import scatter
from tqdm import tqdm

from ito.model import beta_schedule, ema, dpm_solve


class DDPMBase(pl.LightningModule):
    def __init__(
        self,
        score_model_class,
        score_model_kwargs,
        diffusion_steps=1000,
        lr=1e-3,
        beta_scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if beta_scheduler is None:
            beta_min = 1e-4
            beta_max = 0.02
            beta_scheduler = beta_schedule.SigmoidalBetaScheduler(
                diffusion_steps, beta_min, beta_max
            )

        self.beta_scheduler = beta_scheduler
        self.score_model = score_model_class(**score_model_kwargs)



        self.register_buffer("betas", self.beta_scheduler.get_betas())
        self.register_buffer("alphas", self.beta_scheduler.get_alphas())
        self.register_buffer("alpha_bars", self.beta_scheduler.get_alpha_bars())

        self.diffusion_steps = diffusion_steps
        self.lr = lr

        self.ema = ema.ExponentialMovingAverage(
            self.score_model.parameters(), decay=0.99
        )


    def forward(self, *args):
        score = self.score_model(*args)
        return score

    def training_step(self, batch, _):
        loss = self.get_loss(batch)
        self.log("train/loss", loss)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.score_model.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train/loss",
            },
        }

    def _ode_sample(self, batch, forward_callback, ode_steps=100):
        ns = dpm_solve.NoiseScheduleVP(
            "discrete",
            betas=self.betas,
        )

        def t_diff_and_forward(x, t):
            t = t[0]
            batch.t_diff = torch.ones_like(batch.batch) * t
            batch.x = x
            epsilon_hat = forward_callback(batch)
            return epsilon_hat.x

        wrapped_model = dpm_solve.model_wrapper(t_diff_and_forward, ns)
        dpm_solver = dpm_solve.DPM_Solver(wrapped_model, ns)
        batch.x = dpm_solver.sample(batch.x, ode_steps)
        return batch

    def _sample(self, batch, forward_callback):
        batch.x = torch.randn_like(batch.x, device=self.device)

        with torch.no_grad():
            for t in tqdm(torch.arange(self.diffusion_steps - 1, 0, -1)):
                batch.t_diff = torch.ones_like(batch.batch) * t
                epsilon_hat = forward_callback(batch)
                batch.x = self.denoise_sample(t, batch.x, epsilon_hat)

        return batch

    def denoise_sample(self, t, x, epsilon_hat):
        epsilon = torch.randn(x.shape).to(device=x.device)
        preepsilon_scale = 1 / math.sqrt(self.alphas[t])
        epsilon_scale = (1 - self.alphas[t]) / math.sqrt(1 - self.alpha_bars[t])
        post_sigma = math.sqrt(self.betas[t]) * epsilon
        x = preepsilon_scale * (x - epsilon_scale * epsilon_hat.x) + post_sigma

        return x

    def get_noise_img_and_epsilon(self, batch):
        ts = torch.randint(1, self.diffusion_steps, [len(batch)], device=self.device)

        epsilon = self.get_epsilon(batch)

        alpha_bars = self.alpha_bars[ts]
        noise_batch = batch.clone()

        alpha_bars = alpha_bars[batch.batch]

        noise_batch.x = (
            torch.sqrt(alpha_bars) * batch.x.T
            + torch.sqrt(1 - alpha_bars) * epsilon.x.T
        ).T

        noise_batch.t_diff = ts[batch.batch]

        return noise_batch, epsilon

    def get_epsilon(self, batch):
        epsilon = batch.clone()
        epsilon.x = torch.randn(batch.x.shape, device=self.device)
        return epsilon


#  def x_to_geometric(x, batch):
#      batch.clone()
#      batch.x = x
#      return batch
#
#
#  class DDPM(DDPMBase):
#      def sample(self, n_samples, atom_number, ode_steps=0):
#          batch = utils.get_batch_from_atom_number(atom_number, n_samples)
#          batch = batch.to(self.device)
#
#          if ode_steps:
#              return self._ode_sample(
#                  batch, forward_callback=self.forward, ode_steps=ode_steps
#              )
#          return self._sample(batch, forward_callback=self.forward)
#
#      def get_loss(self, batch):
#          noise_batch, epsilon = self.get_noise_img_and_epsilon(batch)
#          epsilon_hat = self.forward(noise_batch)
#          loss = nn.functional.mse_loss(epsilon_hat.x, epsilon.x, reduction="none").sum(
#              -1
#          )
#          loss = scatter(loss, batch.batch, reduce="mean").mean()
#
#          if torch.isnan(loss):
#              raise ValueError("Loss is NaN")
#
#          return loss


class TLDDPM(DDPMBase):
    def sample(self, cond_batch, ode_steps=0):
        cond_batch.to(self.device)

        batch = cond_batch.clone()
        batch.x = torch.randn_like(batch.x, device=self.device)

        def forward_callback(batch):
            return self.forward(batch, cond_batch)

        if ode_steps:
            return self._ode_sample(batch, forward_callback, ode_steps=ode_steps)
        return self._sample(batch, forward_callback=forward_callback)

    def get_loss(self, batch):
        batch_0 = batch["batch_0"]
        batch_t = batch["batch_t"]

        noise_batch, epsilon = self.get_noise_img_and_epsilon(batch_t)
        epsilon_hat = self.forward(noise_batch, batch_0)
        loss = nn.functional.mse_loss(epsilon_hat.x, epsilon.x, reduction="none").sum(
            -1
        )
        loss = scatter(loss, noise_batch.batch, reduce="mean").mean()
        return loss
