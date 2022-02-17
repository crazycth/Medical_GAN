import torchgan
import torch
import torchvision
from torch.utils.data import DataLoader,dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import sampler
import pandas as pd
import os
import cv2
import random


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


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Date(dataset.Dataset):
    def __init__(self,label_dic,train=True,transform_train = None , transform_val = None , root = None):
        self.root = root
        super(Date,self).__init__()
        self.dic = label_dic
        self.train = train
        self.data = os.listdir(self.root)
        random.shuffle(self.data)
        self.len = len(self.data)
        self.transform_train = transform_train
        self.transform_val = transform_val

    def __getitem__(self, index):
        name = self.data[index]
        data = cv2.imread(self.root+"/"+name)
        transform = self.transform_train if self.train else self.transform_val
        data = transform(data)
        label = self.dic[name[:12]]
        return data,label

    def __len__(self):
        return self.len

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
    # transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0,4.0/3.0)),
    # transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_medical_loader(batch_size=16,root="./dataset/pic_save_1"):
    """
    :return: loader_train , loader_val
    """
    data = pd.read_excel("dataset/tem_10.xlsx")
    dic = dict(zip(data['cases'].values, data['tag'].values))
    data_train = Date(dic,True,transform_train,transform_val,root=root)
    #data_val = Date(dic,False,transform_train,transform_val,root=root)
    total_num = len(os.listdir(root))
    train_num = int(total_num)
    loader_train = DataLoader(data_train,batch_size=batch_size,sampler=ChunkSampler(train_num,0),drop_last=True)
    #loader_val = DataLoader(data_val,batch_size=batch_size,sampler=ChunkSampler(val_num,train_num),drop_last=True)
    return loader_train


if __name__ == '__main__':
    loader = get_medical_loader(16)
    print(loader.__iter__().next()[0].shape)
