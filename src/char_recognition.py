'''
Author: Nelson Igbokwe
Project: EMNIST-Classification
'''

import cv2 as cv
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
# from skimage.feature import hog

# =============================================================================
# Classifiers: can be created through ML-Models.ipynb
# =============================================================================
PATH_MODELS='models/CLS_MLP/'
#KNN
# with open('let_KNN_clf.pickle','rb') as f:
#     let_KNN_clf = pk.load(f)
# with open('dig_KNN_clf.pickle','rb') as f:
#     dig_KNN_clf = pk.load(f)
# with open('bal_KNN_clf.pickle', 'rb') as f:
#     bal_KNN_clf = pk.load(f)
# #MLP    
# with open('dig_MLP_clf.pickle','rb') as f:
#     dig_MLP_clf = pk.load(f)
# with open('let_MLP_clf.pickle', 'rb') as f:
#     let_MLP_clf = pk.load(f)
with open(PATH_MODELS+'bal_MLP_clf.pickle','rb') as f:
    bal_MLP_clf = pk.load(f)
#with open('dig_MLP_clf_hog.pickle', 'rb') as f:
#    dig_MLP_clf_hog = pk.load(f)
# =============================================================================
# Image Processing Functions
# =============================================================================
def get_roi_all(image): 
    ''' find objects in image, return vector of segmented images and respective rectangles'''
    # find contours
    image = cv.GaussianBlur(image,(5,5),1)
    ret, image = cv.threshold(image, 90, 255, cv.THRESH_BINARY_INV)
    ctrs, hier = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # find rectangles
    rects = [cv.boundingRect(ctr) for ctr in ctrs]
    #copy each object to image vector
    #rect = [x, y, width, height]
    im_vec = [image[rect[1]-int(rect[3]*0.2):rect[1]+int(rect[3]*1.2), 
                    rect[0]-int(rect[2]*0.2):rect[0]+int(rect[2]*1.2)] for rect in rects]
    return im_vec, rects

def draw_named_rect(image, rectangle, text):
    '''draw rectangle and name it'''
    rect = rectangle
    img = image
    cv.rectangle(img, (rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), 0, 1)
    cv.putText(img, text,(rect[0],rect[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9,0, 1)
    return img
      
def adjust_img(image, width, height):
    '''resize to (height, width)'''
    image = cv.resize(image, dsize = (height,width),interpolation=cv.INTER_AREA)
    image_vec = image.reshape(1,-1)
    return image_vec

def pred_character(classifier, im_vec, target):
    '''classifies object and returns ascii code'''
    #dictionaries to look up class and character depending on target
    balanced_dict = dict(zip([i for i in range(0,47)],
                        [str(i) for i in range(0,10)]+['A','B','C','D','E','F','G',
                                                      'H','I','J','K','L','M','N','O',
                                                      'P','Q','R','S','T','U','V','W',
                                                      'X','Y','Z']+['a','b','d','e','f',
                                                        'g','h','n','q','r','t']))
    letters_dict = dict(zip([i for i in range(1,27)],
                           [chr(n) for n in range(65,91)]))
    # Prediction
    pred  = classifier.predict(np.array(im_vec, 'float64'))
    # set character depending on target_type
    if target== 'digits' or target == 'all':
        pred_char = balanced_dict[pred[0]]
    elif target == 'letters':
        #classes from 1 to 26 (upper + lower case in one class)
        #offset = 64 # A --> 65
        pred_char = letters_dict[pred[0]]
    else:
        raise ValueError('target must be "digits", "all" or "letters"')
    return pred, pred_char

def start_recognition(image_name, clf, target_type):
    '''predict character and draw label on image
    '''
    #read image
    im = cv.imread(image_name,cv.IMREAD_GRAYSCALE)
    # find the character in image
    images_roi, rects = get_roi_all(im)
    for idx, rect in enumerate(rects):
        # fit the image to data set
        try:
            im_vec = adjust_img(images_roi[idx], 28 ,28)
        except Exception:
            #padding in function get_roi_all() might lead to error
            continue
        #predict and return ascii code
        pred, pred_char = pred_character(clf, im_vec, target_type)
        #draw classification
        im = draw_named_rect(im, rect, pred_char)
    return im
# =============================================================================
# GUI
# =============================================================================
class Gui:
    ''' - Display a GUI to allow classification of handwritten characters
        - Use Gui.start() to run GUI; respective classifier and target type as argument needed
    '''
    
    #window + canvas to draw on
    im_w, im_h = (600,600)
    # attributes for calculating accuracy
    __mean = 0
    __tries = 0

    @classmethod
    def __draw(cls, event):
        #draw white line on black background
        offs = cls.__thicknes
        #draws on the canvas and on an invisible image that contains the same drawing
        cls.canvas.create_oval((event.x-offs),(event.y-offs), event.x+offs, event.y+offs, fill='black', outline ='black')
        cls.__draw_Obj.ellipse([(event.x-offs),(event.y-offs), event.x+offs, event.y+offs], fill='black')

    @classmethod
    def __del_img(cls):

        cls.canvas.delete('all')
        #delete img by reinitializing it
        cls.__img = Image.new('L',(cls.im_w, cls.im_h), 255)
        cls.__draw_Obj = ImageDraw.Draw(cls.__img)
        #delete prediction output
        cls.textPred.delete(1.0,tk.END)

    @classmethod
    def __predict(cls):
        '''predict digit,and display classification
        '''
        cls.__img.save('img.png')
        im = start_recognition('img.png', cls.__clf, cls.target_type)
        cv.imshow('prediction', im)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
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
    def __inc_thickness(cls):
        if  cls.__thicknes < 11:
            cls.__thicknes+=1
    
    @classmethod
    def __dec_thickness(cls):
        if  cls.__thicknes > 1:
            cls.__thicknes-=1
    

    @classmethod
    def __key_yes_no(cls, event):
        key = event.char
        if key == 'n' or key == 'N':
            cls.__no()
        if key == 'y' or key == 'Y':
            cls.__yes()
        if key == 'r' or key == 'R':
            cls.__reset()
        if key == '+':
            cls.__inc_thickness()
        if key == '-':
            cls.__dec_thickness()
            

    @classmethod
    def __reset(cls):
        '''reset calculation of accuracy'''
        cls.__mean = 0
        cls.__tries = 0
        cls.entryAcc.delete(0,tk.END)


    @classmethod
    def start(cls, clf, target_type: str):
        if not isinstance(target_type, str):
            raise TypeError('target_type must be of type str')
        # set classifier and target type
        cls.__clf = clf
        cls.target_type = target_type
        #initialize image 'L' = 8Bit, black and white
        cls.__img = Image.new('L',(cls.im_w, cls.im_h),255)
        cls.__draw_Obj = ImageDraw.Draw(cls.__img)
        cls.__thicknes = 5
        m = tk.Tk()
        m.minsize(300,300)

        # canvas to draw on
        cls.canvas = tk.Canvas(m, width=cls.im_w, height=cls.im_h, bg = 'white')
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

if __name__=='__main__':
    model = bal_MLP_clf
    target = 'all' # 'digits, 'letters'
    Gui.start(model, 'all')
