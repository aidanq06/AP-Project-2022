from tkinter import *
from tkinter import ttk

rt = Tk()
frm = ttk.Frame(rt, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World!").grid(column=0, row=0)