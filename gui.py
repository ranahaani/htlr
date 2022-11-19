from keras.models import load_model
from tkinter import *
import tkinter as tk
from lenet.lenet import LeNet
from keras.optimizers import SGD
from PIL import ImageGrab
import numpy as np
import os

opt = SGD(lr=0.01)

model = LeNet.build(numChannels=1, imgRows=28, imgCols=28,
                    numClasses=39,
                    weightsPath=os.path.join("weights.hdf5"))
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


def predict_digit(img):
    print(img)

    img = img.resize((28, 28))
    img = img.convert('L')

    value = np.asarray(img.getdata(), dtype=np.float)
    value = value.flatten()

    img = value.reshape(1, 28, 28, 1)
    img = img / 255.0
    # predicting the class
    res = model.predict([img])[0]

    class_values = {'ب': 0, 'د': 1, 'ی': 2, 'ء': 3, 'ن': 4, 'س': 5, 'ف': 6, 'ش': 7, 'و': 8, 'غ': 9, 'چ': 10, 'ا': 11,
                    'خ': 12, 'ة': 13, 'ڈ': 14, 'ٹ': 15, 'ص': 16, 'ز': 17, 'پ': 18,
                    'ق': 19, 'گ': 20, 'ط': 21, 'م': 22, 'ڑ': 23, 'ذ': 24, 'ع': 25, 'ژ': 26, 'ج': 27, 'N/A': 28, 'ث': 29,
                    'ں': 30, 'ے': 31, 'ظ': 32, 'ل': 33, 'ر': 34, 'ک': 35, 'ض': 36, 'ت': 37, 'ح': 38}

    prediction = np.argmax(res)

    for value, index in class_values.items():
        if index == prediction:
            prediction = value
            print(prediction)
    return prediction, max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(
            self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="NULL..", font=("Helvetica", 40))
        self.classify_btn = tk.Button(
            self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(
            self, text="Clear", command=self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        x = self.winfo_x()
        y = self.winfo_y()
        # Screenshot canvas area
        im = ImageGrab.grab(bbox=(x + 10, y + 70, x + 600, y + 650))

        digit, acc = predict_digit(im)
        self.label.configure(text=str(int(acc * 100)) + '%' + ' : ' + str(digit))

    def draw_lines(self, event):
        global lastx, lasty
        self.x = event.x
        self.y = event.y
        r = 1.5
        self.canvas.create_line(
            (self.x - r, self.y - r, self.x + r, self.y + r), fill='black', width=3, joinstyle=ROUND, smooth=1)


app = App()
mainloop()
lastx, lasty = None, None
