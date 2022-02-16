from tkinter import *
from PIL import Image, ImageTk, ImageDraw
from tkinter import Tk
from tkinter import ttk
from PIL import Image
from pyparsing import White

width, height = 300,300


def save():
    image.save("image.png")
    img = Image.open("image.png")
    resized_img = img.resize((28,28))
    resized_img.save("resized.png")

def draw(arg):
    x,y,x1,y1 = (arg.x-1), (arg.y-1), (arg.x+1), (arg.y+1)
    canvas.create_oval(x,y,x1,y1, fill="white",width=30)
    draw.line([x,y,x1,y1],fill="white",width=30)

app = Tk()

canvas = Canvas(app,bg="white",width=width,height=height)
canvas.pack(expand=YES,fill=BOTH)
canvas.bind("<B1-Motion>", draw)


image = Image.new("RGB", (width,height), (0,0,0))
draw = ImageDraw.Draw(image)

button=Button(text="Save",command=save)
button.pack()

app.mainloop()