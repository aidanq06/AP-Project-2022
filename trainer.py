import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# Download files (requires internet connection)
train = datasets.MNIST('', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=20, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=20, shuffle=False)

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
    for epoch in range(50): 
        for data in trainset:  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.nll_loss(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print(f"{loss} \n Saved! Ran {cout} time(s)")  # more loss is better
        torch.save(net.state_dict(),path)
        cout+=1
        
