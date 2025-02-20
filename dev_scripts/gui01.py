# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 10:31:07 2024

@author: rg5749
"""

import tkinter as tk

# Function to change the label text
def change_text():
    label.config(text="Hello, Tkinter!")

# Create the main window
root = tk.Tk()
root.title("Simple Tkinter GUI")

# Create a label widget
label = tk.Label(root, text="Welcome to Tkinter!")
label.pack(pady=10)

# Create a button widget
button = tk.Button(root, text="Click Me", command=change_text)
button.pack(pady=10)

# Run the main loop
root.mainloop()
