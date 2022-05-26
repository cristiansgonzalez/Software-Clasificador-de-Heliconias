# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:08:27 2020

@author: CristianGonzalez
"""


from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
from tkinter import filedialog
import os
import malla
from PIL import Image, ImageTk 
import PIL 
from skimage import io
#from __future__ import print_function


import numpy as np
import cv2 as cv
from skimage import io
import sys

global filas, columnas

filas=320
columnas=200
#global label1
def buscar():
    
    global nombre,label1    
    
    nombre=filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("tif files","*.tif"),("jpg files","*.jpg"),("png files","*.png"),("all files","*.*")))
    #nombre=filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpg files","*.jpg"),("tif files","*.tif"),("all files","*.*")))
    
    img = Image.open(nombre)
    #print('salida imagen: '+str(img))
    #img2=malla.matriz_interfaz(filas,columnas,1,nombre)
    #io.imsave('redimensinar.TIF', img2*255.0)
    
    
    #cv.imwrite('lenaaaaa.png', img2)
    #im2 = Image.fromarray(img2)
    #im2.save("your_file.jpeg")
    label1.grid_forget()
    img.thumbnail((170, 170), Image.ANTIALIAS)
    #img.thumbnail((96, 170), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label1 = Label(raiz, image=photo)
    label1.grid(column=0, row=1, padx=20)
    label1.img = photo # *
    
    
    #b3=Label(raiz, text=nombre)
    #b3.grid(column=1, row=0, padx=20)

class App():
    print("entro")
    BLUE = [255,0,0]        # rectangle color
    RED = [0,0,255]         # PR BG
    GREEN = [0,255,0]       # PR FG
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    DRAW_PR_BG = {'color' : RED, 'val' : 2}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}

    # setting up flags
    rect = (0,0,1,1)
    drawing = False         # flag for drawing curves
    rectangle = False       # flag for drawing rect
    rect_over = False       # flag to check if rect drawn
    rect_or_mask = 100      # flag for selecting rect or mask mode
    value = DRAW_FG         # drawing initialized to FG
    thickness = 3           # brush thickness

    def onmouse(self, event, x, y, flags, param):
        # Draw Rectangle
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x,y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # draw touchup curves

        if event == cv.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

    def run(self):
        # Loading images
        global res,sal2,sal3
        #global img
        if len(sys.argv) == 2:
            filename = sys.argv[1] # for drawing purposes
        else:
            print("No input image given, so loading default image, lena.jpg \n")
            print("Correct Usage: python grabcut.py <filename> \n")
            
            #filename =img2
            #filename = nombre
            #filename = 'lena.jpg'

        #self.img = cv.imread(cv.samples.findFile(filename))
        
        self.img=malla.matriz_interfaz(filas,columnas,1,nombre)
        #self.img=self.img.convert('RGB')
        self.img=cv.convertScaleAbs(self.img)
        self.img2 = self.img.copy()                               # a copy of original image
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
        self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown

        # input and output windows
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.img.shape[1]+10,90)

        print(" Instructions: \n")
        print(" Draw a rectangle around the object using right mouse button \n")

        while(1):

            cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)

            # key bindings
            if k == 27:         # esc to exit
                break
            elif k == ord('0'): # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = self.DRAW_BG
            elif k == ord('1'): # FG drawing
                print(" mark foreground regions with left mouse button \n")
                self.value = self.DRAW_FG
            elif k == ord('2'): # PR_BG drawing
                self.value = self.DRAW_PR_BG
            elif k == ord('3'): # PR_FG drawing
                self.value = self.DRAW_PR_FG
            elif k == ord('s'): # save image
                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                res = np.hstack((self.img2, bar, self.img, bar, self.output))
                cv.imwrite('grabcut_output.png', res)
                
                
                sal2=np.zeros((filas-1,columnas-1,3))
                sal2=res[0:filas-1,(columnas*2)+10:(columnas*3)+10]
                
                cv.imwrite('segmentada.png', sal2)
                sal3=io.imread('segmentada.png')
                
                
                print(" Result saved as image \n")
                break
            elif k == ord('r'): # reset everything
                print("resetting \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                self.img = self.img2.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
                self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown
            elif k == ord('n'): # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                try:
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    if (self.rect_or_mask == 0):         # grabcut with rect
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif (self.rect_or_mask == 1):       # grabcut with mask
                        cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()

            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)

        print('Done')

def seg():
    
    global label2
    
    App().run()
    cv.destroyAllWindows()
    
    label2.grid_forget()
    img = Image.open('segmentada.png')
       
    #io.imsave('redimensinar.TIF', img2*255.0)
    
    
    #cv.imwrite('lenaaaaa.png', img2)
    #im2 = Image.fromarray(img2)
    #im2.save("your_file.jpeg")
    
    
    img.thumbnail((170, 170), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label2 = Label(raiz, image=photo)
    label2.grid(column=1, row=1, padx=20)
    label2.img = photo # *
    #b5=Label(raiz, text=os.getcwd())
    #b5.grid(column=1, row=1, padx=20)

def clasificar():
    
    global pred, label3
    #pred=malla.prueba_interfaz_intersecion(sal3)
    #pred=malla.prueba_interfaz_ANN(sal3)
    pred=malla.prueba_interfaz_KNN(sal3)
    #print(pred)
    
    label3.grid_forget()
    
    if pred[0]==0:
        
        img = Image.open('0.png')
        img.thumbnail((170, 170), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label3 = Label(raiz, image=photo)
        label3.grid(column=2, row=1, padx=20)
        label3.img = photo # *
        
    if pred[0]==1:
        
        img = Image.open('1.png')
        img.thumbnail((170, 170), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label3 = Label(raiz, image=photo)
        label3.grid(column=2, row=1, padx=20)
        label3.img = photo # *
        
    if pred[0]==2:
        
        img = Image.open('2.png')
        img.thumbnail((170, 170), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label3 = Label(raiz, image=photo)
        label3.grid(column=2, row=1, padx=20)
        label3.img = photo # * 
        
    if pred[0]==3:
        
        img = Image.open('3.png')
        img.thumbnail((170, 170), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label3 = Label(raiz, image=photo)
        label3.grid(column=2, row=1, padx=20)
        label3.img = photo # *
    
    
    



raiz = Tk()
raiz.geometry('600x270') # anchura x altura
#raiz.configure(bg = 'white')
raiz.configure(bg = 'beige')
raiz.iconbitmap('favicon.ico')
raiz.title('Clasificador de Heliconias')



img = Image.open('blanco.png')
img.thumbnail((170, 170), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(img)
label1 = Label(raiz, image=photo)
label1.grid(column=0, row=1, padx=20)
label1.img = photo # *

photo = ImageTk.PhotoImage(img)
label2 = Label(raiz, image=photo)
label2.grid(column=1, row=1, padx=20)
label2.img = photo # *

photo = ImageTk.PhotoImage(img)
label3 = Label(raiz, image=photo)
label3.grid(column=2, row=1, padx=20)
label3.img = photo # *

raiz.resizable(0,0)
# Define un botón en la parte inferior de la ventana
# que cuando sea presionado hará que termine el programa.
# El primer parámetro indica el nombre de la ventana 'raiz'
# donde se ubicará el botón

#b1=Button(raiz, text='Salir', width=10,command=raiz.destroy)#.pack(side=BOTTOM)
#b1.grid(column=0, row=4, padx=20)
#b1.place(x=20,y=70)

#raiz.nombre=


examinar2 = PhotoImage(file = "examinar2.png") 
b2=Button(raiz, text = 'Examinar...',bd=-1,image=examinar2,command=buscar)
#b2=Button(raiz, text='Abrir...', width=10,command=buscar)
b2.grid(column=0, row=0, padx=30,pady=20)

segmentar2 = PhotoImage(file = "segmentar2.png") 
b4=Button(raiz, text='Segmentar',bd=-1, image=segmentar2,command=seg)
#b4=Button(raiz, text='Segmentar', width=10,command=seg)
b4.grid(column=1, row=0, padx=20)

clasificar2 = PhotoImage(file = "clasificar2.png")
b5=Button(raiz, text='Clasificar',bd=-1, image=clasificar2,command=clasificar)
b5.grid(column=2, row=0, padx=30)



#h1=malla.matriz_interfaz(320,120,1,nombre)
#h2=cv.convertScaleAbs(h1)



raiz.mainloop()




    
    
    
    
    
    
    
    
    
    
    