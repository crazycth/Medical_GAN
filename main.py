import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,dataset
from torch.utils.data import sampler,TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2
import torchvision
import torchgan.models
from my_model import *
from DataLoader import *
from torchgan.losses import *
from torchgan.trainer import *
dtype = torch.float32

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # Use deterministic cudnn algorithms
    torch.backends.cudnn.deterministic = True
    epochs = 60
else:
    device = torch.device("cpu")
    epochs = 5

print("Device: {}".format(device))
print("Epochs: {}".format(epochs))

loader = get_medical_loader(8)
#print(loader.__iter__().next()[0].shape)
cgan = dcgan_network()
lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]
trainer_cgan = Trainer(
    cgan, lsgan_losses, sample_size=64, epochs=epochs, device=device
)
trainer_cgan(loader)