# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:02:12 2020

@author: shoun
"""

#from keras.models import model_from_json
from keras.models import load_model
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("videos/facial_exp.mkv")


loaded_model = load_model('face_expression_model.h5')

EMOTIONS_LIST = {0:"angry",
                 1:"disgust",
                 2:"fear",
                 3:"happy",
                 4:"neutral",
                 5:"sad",
                 6:"surprise"}

def mapper(val):
    
    return EMOTIONS_LIST[val]

width = cap.set(3,1080)
height = cap.set(4,1920)

while True :
    
    ret , frame = cap.read()
    
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray ,1.3,5)
    
    for (x,y,w,h) in faces :
        
    
        cv2.rectangle(frame , (x,y) , (x+w , y+h), (0,255,0) , 2)
        
        roi_gray = gray[y:y+h , x:x+w]
        
        roi = cv2.resize(roi_gray , (46,46))
        pred = loaded_model.predict(roi[np.newaxis,:,:,np.newaxis]) 
        move_code = np.argmax(pred[0])
        move_name = mapper(move_code)
        
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, move_name, (x, y), font, 1, (255, 255, 0), 2)
         
        cv2.imshow('frame',frame)
            
    if cv2.waitKey(15) & 0xFF == 27:
        break
    
cap.release() 
cv2.destroyAllWindows()   
            
            
            
        