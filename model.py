from torchgan.models import *
import torch
def dcgan_network():
    dcgan_network = {
        "generator": {
            "name": DCGANGenerator,
            "args": {
                "encoding_dims": 100,
                "out_channels": 1,
                "step_channels": 32,
                "nonlinearity": torch.nn.LeakyReLU(0.2),
                "last_nonlinearity": torch.nn.Tanh(),
            },
            "optimizer": {"name": torch.optim.Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
        },
        "discriminator": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": 1,
                "step_channels": 32,
                "nonlinearity": torch.nn.LeakyReLU(0.2),
                "last_nonlinearity": torch.nn.LeakyReLU(0.2),
            },
            "optimizer": {"name": torch.optim.Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
        },
    }
    return dcgan_network