import torch


class BetaSchedulerBase:
    def __init__(self, diffusion_steps, beta_min=None, beta_max=None):
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_alpha_bars(self):
        betas = self.get_betas()
        alpha_bars = []
        alpha_bar = 1
        for beta in betas:
            alpha_bars.append(alpha_bar)
            alpha = 1 - beta
            alpha_bar = alpha * alpha_bar

        return torch.Tensor(alpha_bars)

    def get_alphas(self):
        return 1 - self.get_betas()

    def get_snr_weight(self):
        return self.get_snr()[:-1] - self.get_snr()[1:]

    def get_betas(self):
        raise NotImplementedError

    def get_snr(self):
        alpha_bars_squared = self.get_alpha_bars() ** 2
        sigma_bars_squared = 1 - alpha_bars_squared

        snr = alpha_bars_squared / sigma_bars_squared
        return snr


class LinearBetaScheduler(BetaSchedulerBase):
    def get_betas(self):
        betas = [
            self.beta_min + (t / self.diffusion_steps) * (self.beta_max - self.beta_min)
            for t in range(self.diffusion_steps)
        ]
        return torch.Tensor(betas)


class SigmoidalBetaScheduler(BetaSchedulerBase):
    def get_betas(self):
        ts = torch.linspace(-8, -4, self.diffusion_steps)
        betas = torch.sigmoid(ts)
        return betas
