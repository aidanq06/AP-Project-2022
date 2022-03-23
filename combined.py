import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tkinter import *
from PIL import Image, ImageDraw
from tkinter import Tk
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

def save():
    image.save("image.png")
    img = Image.open("image.png")
    resized_img = img.resize((28,28))
    resized_img.save("resized.png")

def evaluate():
    save()
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
            
            widget = Label(canvas, text=f'Predicted: {torch.argmax(i[1])}', fg='black', bg='white')
            widget.place(x=5,y=280)

            print(torch.argmax(i[1]),) # Should print out what number it thinks.
            break

def draw(arg):
    x,y,x1,y1 = (arg.x-1), (arg.y-1), (arg.x+1), (arg.y+1)
    canvas.create_oval(x,y,x1,y1, fill="white",width=30)
    draw.line([x,y,x1,y1],fill="white",width=30)
    evaluate()

def clear():
    canvas.delete("all")
    draw.rectangle((0,0,500,500),"black")
    save()

width, height = 300,300
app = Tk()

canvas = Canvas(app,bg="white",width=width,height=height)
canvas.pack(expand=YES,fill=BOTH)
canvas.bind("<B1-Motion>", draw)


image = Image.new("RGB", (width,height), (0,0,0))
draw = ImageDraw.Draw(image)

button=Button(text="Evaluate",command=evaluate)
button.pack()
button=Button(text="Clear",command=clear)
button.pack()


app.mainloop()

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







