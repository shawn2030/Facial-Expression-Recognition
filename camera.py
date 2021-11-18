# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:31:27 2021

@author: shoun
"""

from keras.models import load_model
import numpy as np
import cv2
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model('face_expression_model.h5')

EMOTIONS_LIST = {0:"angry",
                 1:"disgust",
                 2:"fear",
                 3:"happy",
                 4:"neutral",
                 5:"sad",
                 6:"surprise"}

def mapper(val):
    
    return EMOTIONS_LIST[val]


class VideoCamera(object):
    
    def __init__(self):
        self.video = cv2.VideoCapture("videos/facial_exp.mkv")
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        ret , frame = self.video.read()
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray ,1.3 ,5)
    
        for (x,y,w,h) in faces :
        
            cv2.rectangle(frame , (x,y) , (x+w , y+h), (0,255,0) , 2)
            
            roi_gray = gray[y:y+h , x:x+w]
            
            roi = cv2.resize(roi_gray , (46,46))
            pred = model.predict(roi[np.newaxis,:,:,np.newaxis]) 
            move_code = np.argmax(pred[0])
            move_name = mapper(move_code)
            
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, move_name, (x, y), font, 1, (255, 255, 0), 2)   
            
        _ , jpeg = cv2.imencode('.jpg',frame)

        return jpeg.tobytes()         
        

    