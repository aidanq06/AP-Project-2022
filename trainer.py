import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# A lot of this code was taken from
# https://pythonprogramming.net/building-deep-learning-neural-network-pytorch/

# Download files (requires internet connection)
train = datasets.MNIST('', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

path = "model/training.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net) # Prints the neural net layout

model = Net()
model.load_state_dict(torch.load(path))
model.eval()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

cout = int()
class Trainer():
    
    # Changing the range will affect the accuracy of the program.
    for epoch in range(1000): 
        for data in trainset:
            X, y = data  # x = batch of features, y = batch of targets.
            net.zero_grad()
            output = net(X.view(-1,784))  
            loss = F.nll_loss(output, y)  
            loss.backward() 
            optimizer.step()  
        print(f"{loss} \n Saved! Ran {cout} time(s)")  # more loss is better
        torch.save(net.state_dict(),path)
        cout+=1
        
