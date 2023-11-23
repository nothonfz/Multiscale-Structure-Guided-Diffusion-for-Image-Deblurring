from model.sr3_modules.diffusion import GaussianDiffusion
from model.sr3_modules.unet import UNet as UNet
import torch.nn as nn

class DVSr(nn.Module):
    def __init__(self, img_size, device, loss_type, use_noisy, use_lr, lr_rate):
        super().__init__()
        assert not (use_noisy and use_lr)
        self.diffusion = GaussianDiffusion(UNet(inner_channel=64), img_size, conditional=True
                                           , loss_type=loss_type, use_noisy=use_noisy, use_lr=use_lr, lr_rate=lr_rate)
        self.device = device

    def forward(self, blur, sharp):
        schedule = {"schedule": "linear",
                    "n_timestep": 2000,
                    "linear_start": 1e-6,
                    "linear_end": 1e-2}
        self.diffusion.set_new_noise_schedule(schedule, self.device)
        self.diffusion.set_loss(self.device)
        l_pix = self.diffusion(sharp, blur)
        b, c, h, w = blur.shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        return l_pix

