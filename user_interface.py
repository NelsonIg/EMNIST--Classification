#!/usr/bin/env python
# coding: utf-8

# # GUI

# In[1]:


import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import cv2 as cv
import pickle as pk


# In[2]:


class Gui:
    ''' - Display a GUI to allow classification of handwritten characters
        - Use start() to run GUI, respective classifier as argument needed
    '''
    
    #window + canvas to draw on
    im_w, im_h = (224,224)
    # invisible image for copying canvas
    __img = Image.new('L',(im_w, im_h))
    __draw = ImageDraw.Draw(__img)
    #canvas
    canvas = None
    #text windows
    textPred = None
    entryAcc = None

    # attributes for calculating accuracy
    __mean = 0
    __tries = 0
    # Classifier
    __clf = None

    @classmethod
    def __draw(cls, event):
        #draw white line on black background
        offs = 10
        #draws on the canvas and on an invisible image that contains the same drawing
        cls.canvas.create_oval((event.x-offs),(event.y-offs), event.x+offs, event.y+offs, fill='white', outline ='white')
        cls.__draw.ellipse([(event.x-offs),(event.y-offs), event.x+offs, event.y+offs], fill='white')

    @classmethod
    def __del_img(cls):

        cls.canvas.delete('all')
        #delete img by reinitializing it
        cls.__img = Image.new('L',(cls.im_w, cls.im_h))
        cls.__draw = ImageDraw.Draw(cls.__img)
        #delete prediction output
        cls.textPred.delete(1.0,tk.END)

    @classmethod
    def __predict(cls):
        '''predict digit,and display classification
        '''
        cls.__img.save('img.png')
        im = cv.imread('img.png',cv.IMREAD_GRAYSCALE)
        im = cv.GaussianBlur(im,(5,5),1)
        im = cv.resize(im, dsize = (28,28),interpolation=cv.INTER_AREA)
        im = im.reshape(1,-1)
        pred  = cls.__clf.predict(im)
        #print prediction
        if pred<= 9:
            offset = 48 # 0--> 48
        elif pred <36:
            offset = 55 # A --> 65
        else:
            offset = 60# a --> 97
        cls.textPred.delete(1.0,tk.END)
        cls.textPred.insert(tk.INSERT,f'class: {pred[0]}\nascii: '+chr(offset + pred[0]))

    #detecting the mean of user digits in percent   
    @classmethod
    def __yes(cls):
        '''Correct classification, adjust accuracy'''
        cls.__tries +=1
        cls.__mean = cls.__mean*(cls.__tries-1)/cls.__tries + 1/cls.__tries
        cls.entryAcc.delete(0,tk.END)
        cls.entryAcc.insert(tk.INSERT,str(round(cls.__mean*100,1))+'%')
        cls.__del_img()

    @classmethod
    def __no(cls):
        '''wrong classification, adjust accuracy'''
        cls.__tries +=1
        cls.__mean = cls.__mean*(cls.__tries-1)/cls.__tries
        cls.entryAcc.delete(0,tk.END)
        cls.entryAcc.insert(tk.INSERT,str(round(cls.__mean*100,1))+'%')
        cls.__del_img()

    @classmethod
    def __key_yes_no(cls, event):
        key = event.char
        if key == 'n' or key == 'N':
            cls.__no()
        if key == 'y' or key == 'Y':
            cls.__yes()
        if key == 'r' or key == 'R':
            cls.__reset()

    @classmethod
    def __reset(cls):
        '''reset calculation of accuracy'''
        cls.__mean = 0
        cls.__tries = 0
        cls.entryAcc.delete(0,tk.END)


    @classmethod
    def start(cls, clf):
        # set classifier
        cls.__clf = clf
        m = tk.Tk()
        m.minsize(300,300)

        # canvas to draw on
        cls.canvas = tk.Canvas(m, width=cls.im_w, height=cls.im_h, bg = 'black')
        cls.canvas.grid(row = 0, column = 1,columnspan=5 ,rowspan = 5)
        cls.canvas.x_old = None
        cls.canvas.y_old = None
        cls.canvas.x_new = None
        cls.canvas.y_new = None
        cls.canvas.winfo_height
        m.bind('<B1-Motion>', cls.__draw)

        #Buttons
        buttonDelete = tk.Button(m, text='delete',width=10,height = 1,command=cls.__del_img)
        buttonDelete.grid(row = 0, column = 0)

        buttonPred = tk.Button(m, text='predict',width=10,height = 1,command=cls.__predict)
        buttonPred.grid(row = 1, column = 0)
        labelPred =tk.Label(m, text='Prediction:')
        labelPred.grid(row = 2, column = 0)
        cls.textPred = tk.Text(m,width = 10, height = 2, font = 15)
        cls.textPred.grid(row = 3, column = 0)

        buttonYes = tk.Button(m, text='Yes [y]',width=10,height = 1,bg ='green', command=cls.__yes)
        buttonYes.grid(row = 4, column = 0)
        buttonNo = tk.Button(m, text='No [n]',width=10,height = 1,bg ='red', command=cls.__no)
        buttonNo.grid(row = 5, column = 0)

        #bind 'n' and 'y' keyboard
        m.bind('<Key>', cls.__key_yes_no)


        labelAcc =tk.Label(m, text='Accuracy:')
        labelAcc.grid(row = 6, column = 0)
        cls.entryAcc = tk.Entry(m,width = 10, font = 15)
        cls.entryAcc.grid(row = 7, column = 0)
        buttonReset = tk.Button(m, text='Reset % [r]',width=10,height = 1,bg ='grey', command=cls.__reset)
        buttonReset.grid(row = 7, column = 1)


        m.mainloop()
        


# In[ ]:




