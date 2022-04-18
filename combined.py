# Imports
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from tkinter import *
from PIL import Image, ImageDraw

values = list()

# init of the neural net
# DON'T INCLUDE TRAINING HERE
# Training is done with the trainer.py file
class Net(nn.Module):

    # make sure to include the fact that this was taken from pytorch documentation ###########################
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    # make sure to include the fact that this was taken from pytorch documentation######################################
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Saves the canvas and converts it into 28x28, so it is readable by the program.
def save():
    image.save("image.png")
    img = Image.open("image.png")
    resized_img = img.resize((28,28))
    resized_img.save("resized.png")

# Takes the 28x28 png file and evaluates it. It imports an existing neural net model that was 
# created by the 'trainer.py' file
def evaluate():
    save()
    img = Image.open("resized.png")
    path = "model/training.pt"


    model = Net()
    model.load_state_dict(torch.load(path))
    model.eval()
    # test

    transform = transforms.ToTensor()
    tensor_array = transform(img) # Tensor

    with torch.no_grad():
        x = tensor_array
        output = model(x.view(-1,784))
        for i in enumerate(output):
            
            widget = Label(canvas, text=f'Predicted: {torch.argmax(i[1])}', fg='black', bg='white')
            widget.place(x=5,y=280)
            print(torch.argmax(i[1])) # Should print out what number it thinks.

            # Saves a list of all predicted numbers for the "info" function
            global values
            values = i[1].tolist()
            
            break

# Drawing function, also auto-evaluates
def draw(arg):
    x,y,x1,y1 = (arg.x-1), (arg.y-1), (arg.x+1), (arg.y+1)
    canvas.create_oval(x,y,x1,y1, fill="white",width=30)
    draw.line([x,y,x1,y1],fill="white",width=30)
    evaluate() # Can remove so that it doesn't evaluate for each new pixel drawn

# Draws a (white) black rectangle over the entire canvas, erasing all contents.
def clear():
    canvas.delete("all")
    draw.rectangle((0,0,500,500),"black")
    save()

# More Info Button
# Returns extra info on the evaluation. Returns other possibilities.
def info():
    fixed = dict()
    final = dict()
    # rounds each neuron value
    for i in range(len(values)):
        temp = round(values[i],2)
        values[i] = temp    

    # assigns numeric value to appropriate value
    # returns a dict {"0.1231" = 1}
    for i in range(len(values)):
        ind = values[i]
        fixed[f"{ind}"] = i
    
    # sort the list greatest -> least
    values.sort(reverse=True)
    

    for i in values:
        temp = fixed[f"{i}"]

        final[i]=temp

    #print(final,values)
    print("Values closest to 0 represent its confidence.\n")
    for i in final:
        print(f'"{final[i]}" -> {i}')
    
# Declarations of the GUI
width, height = 300,300
app = Tk()

canvas = Canvas(app,bg="white",width=width,height=height)
canvas.pack(expand=YES,fill=BOTH)
canvas.bind("<B1-Motion>", draw)


image = Image.new("RGB", (width,height), (0,0,0))
draw = ImageDraw.Draw(image)

button=Button(text="More Info",command=info)
button.pack()
button=Button(text="Clear",command=clear)
button.pack()

app.mainloop() # Make sure to always include this at end, GUIs cannot function without it.