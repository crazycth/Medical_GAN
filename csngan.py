import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock
from torch_mimicry.nets.sngan import sngan_base

from torch_mimicry.nets.gan import gan


class SNGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, bottom_width, loss_type='hinge', **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)
        
    def generate_images(self, num_images, c=None, device=None):
        r"""
        Generate images with possibility for conditioning on a fixed class.
        Args:
            num_images (int): The number of images to generate.
            c (int): The class of images to generate. If None, generates random images.
            device (int): The device to send the generated images to.
        Returns:
            tuple: Batch of generated images and their corresponding labels.
        """
        if device is None:
            device = self.device

        if c is not None and c >= self.num_classes:
            raise ValueError(
                "Input class to generate must be in the range [0, {})".format(
                    self.num_classes))

        if c is None:
            fake_class_labels = torch.randint(low=0,
                                              high=self.num_classes,
                                              size=(num_images, ),
                                              device=device)

        else:
            fake_class_labels = torch.randint(low=c,
                                              high=c + 1,
                                              size=(num_images, ),
                                              device=device)

        noise = torch.randn((num_images, self.nz), device=device)
        fake_images = self.forward(noise, fake_class_labels)

        return fake_images
        
class ConditionSNGANGenerator128(SNGANBaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, num_classes=2, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)
        self.num_classes = num_classes
        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True, num_classes=2)
        self.block3 = GBlock(self.ngf, self.ngf >> 1, upsample=True, num_classes=2)
        self.block4 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True, num_classes=2)
        self.block5 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True, num_classes=2)
        self.block6 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True, num_classes=2)
        self.b7 = nn.BatchNorm2d(self.ngf >> 4)
        self.c7 = nn.Conv2d(self.ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)

    def forward(self, x, y=None):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        if y == None:
            # y = torch.randint(0,2,(1,)).to(device)
            y = torch.Tensor((1,)).to(device).long()
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.block6(h, y)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.c7(h))

        return h


class SNGANGenerator128(sngan_base.SNGANBaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block4 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block5 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        self.block6 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
        self.b7 = nn.BatchNorm2d(self.ngf >> 4)
        self.c7 = nn.Conv2d(self.ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = h.long()
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.c7(h))

        return h


class SNGANDiscriminator128(sngan_base.SNGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=1024, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf >> 4)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        self.block6 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l7 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l7.weight.data, 1.0)

        self.embed = nn.Embedding(2, (self.ndf>>2) * 2)
    def forward(self, x,y):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        # h = x.long()
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        label = self.embed(y)
        h = self.concat(label,h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)

        return output

import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan
from DataLoader import *
# !rm -rf /data/Medical/gan/log
# Data handling objects
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

train_dataloader = get_medical_loader(32,r"dataset/pic_save_128")

# Define models and optimizers
netG = ConditionSNGANGenerator128().to(device)
netD = sngan.SNGANDiscriminator128().to(device)
optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

# Start training
trainer = mmc.training.Trainer(
    netD=netD,
    netG=netG,
    optD=optD,
    optG=optG,
    n_dis=5,
    num_steps=100000,
    lr_decay='linear',
    dataloader=train_dataloader,
    log_dir='./log_csngan/example',
    device=device,
    vis_steps=400,
    )
trainer.train()