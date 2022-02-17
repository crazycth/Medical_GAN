from torchgan.models import *
import torch
from torch import nn
from torch.optim import Adam

def dcgan_network():
    dcgan_network = {
        "generator": {
            "name": DCGANGenerator,
            "args": {
                "encoding_dims": 100,
                "out_size":32,
                "out_channels": 3,
                "step_channels": 64,
                "nonlinearity": torch.nn.LeakyReLU(0.2),
                "last_nonlinearity": torch.nn.Tanh(),
            },
            "optimizer": {"name": torch.optim.Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
        },
        "discriminator": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": 3,
                "in_size":32,
                "step_channels": 32,
                "nonlinearity": torch.nn.LeakyReLU(0.2),
                "last_nonlinearity": torch.nn.LeakyReLU(0.2),
            },
            "optimizer": {"name": torch.optim.Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
        },
    }
    return dcgan_network


def cgan():
    cgan_network = {
        "generator": {
            "name": ConditionalGANGenerator,
            "args": {
                "encoding_dims": 100,
                "num_classes": 2,  # MNIST digits range from 0 to 9
                "out_channels": 3,
                "step_channels": 32,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
        },
        "discriminator": {
            "name": ConditionalGANDiscriminator,
            "args": {
                "num_classes": 10,
                "in_channels": 3,
                "step_channels": 32,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
        },
    }
    return cgan_network