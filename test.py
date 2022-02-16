
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy
import PIL
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

img = Image.open("resized.png")
train = datasets.MNIST('', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

def test1():
    with torch.no_grad():
        for data in testset:
            print(data)

def transformer():
    transform = transforms.ToTensor()
    tensor_array = transform(img)
    print(tensor_array.view(-1,784))

    plt.imshow(img)
    plt.show()

transformer()