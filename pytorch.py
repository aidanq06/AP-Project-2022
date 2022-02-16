import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

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


img = Image.open("resized.png")
path = "model/training.pt"

model = Net()
model.load_state_dict(torch.load(path))
model.eval()
# test

transform = transforms.ToTensor()
tensor_array = transform(img) # this is a tensor

with torch.no_grad():
    x = tensor_array
    output = model(x.view(-1,784))
    for i in enumerate(output):
        print(torch.argmax(i[1]),) # Should print out what number it thinks.
        break

"""
with torch.no_grad():
    for data in testset:
        print(data)
        x, y = data
        print(f"X:{x}\n",f"Y:{y}\n")
        output = net(x.view(-1,784))
        for idx, i in enumerate(output):
            print(f"IDX:{idx}\n",f"I:{i}\n")
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1"""

"""print(f"Accuracy: {round(correct/total, 3)*100}% (In terms of 0 - 1 value) {round(correct/total, 3)}")
plt.imshow(X[random.randint(1,9)].view(28,28))
plt.show()
"""
