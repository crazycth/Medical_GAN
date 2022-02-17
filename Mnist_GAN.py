import torchgan
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import *
import numpy as np
from DataLoader import *
from torchgan.losses import *
from torchgan.trainer import Trainer
from my_model import *


minimax_losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
wgangp_losses = [
    WassersteinGeneratorLoss(),
    WassersteinDiscriminatorLoss(),
    WassersteinGradientPenalty(),
]


lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # Use deterministic cudnn algorithms
    torch.backends.cudnn.deterministic = True
    epochs = 10
else:
    device = torch.device("cpu")
    epochs = 5

print("Device: {}".format(device))
print("Epochs: {}".format(epochs))

trainer = Trainer(
    dcgan_network(), lsgan_losses, sample_size=64, epochs=epochs, device=device
)



mnist_loader = Mnist_loader(64,True)
trainer(mnist_loader)

# # Grab a batch of real images from the dataloader
# real_batch = next(iter(mnist_loader))
#
# # Plot the real images
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.axis("off")
# plt.title("Real Images")
# plt.imshow(
#     np.transpose(
#         torchvision.utils.make_grid(
#             real_batch[0].to(device)[:64], padding=5, normalize=True
#         ).cpu(),
#         (1, 2, 0),
#     )
# )
#
# # Plot the fake images from the last epoch
# plt.subplot(1, 2, 2)
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(plt.imread("{}/epoch{}_generator.png".format(trainer.recon, trainer.epochs)))
# plt.show()