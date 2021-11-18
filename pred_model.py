# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:34:41 2021

@author: shoun
"""

import tensorflow as tf
from keras.models import model_from_json
import numpy as np

class FacialExpressionModel(object):
    
    EMOTIONS_LIST = {0:"angry",
                 1:"disgust",
                 2:"fear",
                 3:"happy",
                 4:"neutral",
                 5:"sad",
                 6:"surprise"}
    
    def __init__(self,model_json_file,model_weights_file):
        
        #reading the json model file so that we could import our model framework here
        with open(model_json_file,"r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        
        #loading the weights in our newlyu created model framework which now after adidng weights wil be a working facial expression model
        self.loaded_model.load_weights(model_weights_file)
        
        #what is make_predict_function()
        self.loaded_model.make_predict_function()
        
    def predict_emotion(self,img):
        
        self.preds = self.loaded_model.predict(img)

        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]        