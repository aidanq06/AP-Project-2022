from tkinter import *
from PIL import Image, ImageTk, ImageDraw
from tkinter import Tk
from tkinter import ttk
from PIL import Image
from pyparsing import White

width, height = 400,400


def save():
    image.save("image.png")

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