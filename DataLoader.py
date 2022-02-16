import torchgan
import torch
import torchvision
from torch.utils.data import DataLoader,dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def Mnist_loader(batch_size,shuffle=True):
    dataset = dsets.MNIST(
        root="./dataset",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,),std=(0.5,)),
            ]
        ),
        download=True
    )
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    loader = Mnist_loader(64,True)
    print(loader.__iter__().next()[0].shape)


