import torch
import torch.nn as nn
from torch.nn import utils
from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock
from torch_mimicry.nets.sngan import sngan_base

from torch_mimicry.nets.gan import gan


class ConditionSNGANGenerator128(sngan_base.SNGANBaseGenerator):
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

    def forward(self, x, y):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        if y == None:
            batch_size = x.shape[0]
            y = torch.tensor((0,)*batch_size).to(device)
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
    
    def generate_images_with_labels(self, num_images, c=None, device=None):
        r"""
        Generate images with possibility for conditioning on a fixed class.
        Additionally returns labels.
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

        return fake_images, fake_class_labels
    
    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and labels
        fake_images, fake_class_labels = self.generate_images_with_labels(
            num_images=batch_size, device=device)
        
        # Compute output logit of D thinking image real
        output = netD(fake_images, fake_class_labels)

        # Compute loss and backprop
        errG = self.compute_gan_loss(output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data
    





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
        self.block4 = DBlock((self.ndf >> 2) + 512, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        self.block6 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l7 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l7.weight.data, 1.0)

        # self.embed = nn.Embedding(2, (self.ndf>>2) * 2)
        self.l_y = utils.spectral_norm(nn.Embedding(2, 512))
        
    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):

        self.zero_grad()
        real_images, real_labels = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter
        # Produce logits for real images
        output_real = self.forward(real_images, real_labels)

        # Produce fake images
        fake_images, fake_labels = netG.generate_images_with_labels(num_images=batch_size,
                                           device=device,c=0)
        fake_images.detach()
        fake_labels.detach()
        # Produce logits for fake images
        output_fake = self.forward(fake_images, fake_labels)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        # Backprop and update gradients
        errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD.item(), group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data
    
    def forward(self, x, y):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None:
            emb = self.l_y(y).unsqueeze(-1).unsqueeze(-1)
            emb = emb.expand(emb.size(0), emb.size(1), h.size(2), h.size(3))
            h = torch.cat((h, emb), dim=1)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)

        #output = self.l7(h) + torch.dot(h,y)
        
        return output


import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan
from Dataloader import *
# Data handling objects
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

train_dataloader = get_loader(32,r"pic_trans_128")

# Define models and optimizers
netG = ConditionSNGANGenerator128().to(device)
netD = SNGANDiscriminator128().to(device)
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
    log_dir='./log/reality',
    device=device,
    vis_steps=200,
    )
trainer.train()
