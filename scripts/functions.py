# -*- coding: utf-8 -*-
"""
Created on Sun May  7 07:17:30 2017

@author: ksevta
"""
import numpy as np
import pandas as pd
import cv2
import glob
    
def load_data(width,height,depth):
    
    path = './data/train/**/*.jpg'
    img_data = []
    i = 0
    Type = []
    print('[info] loading images...')
    for filename in glob.glob(path):
        img = cv2.imread(filename)   
        if img.size != None:
            if depth == 1:                
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img,(width,height), interpolation = cv2.INTER_LINEAR)
            #img = img.flatten()            
            img_data.append(img)
            s = 'Type'+(list((filename.split('/'))[3]))[5]
            Type.append(s)
            i += 1
    file1 = './data store/data'+str(width)+'x'+str(height)+'x'+str(depth) 
    file2 = './data store/labels'+str(width)+'x'+str(height)+'x'+str(depth) 
    np.save(file1,img_data) 
    np.save(file2,Type)  
    return 
    

    
