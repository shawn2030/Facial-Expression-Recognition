# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:48:25 2020

@author: shoun
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from livelossplot import PlotLossesKeras
from livelossplot.keras import PlotLossesCallback



from keras.preprocessing.image import ImageDataGenerator
datagen_train = ImageDataGenerator(horizontal_flip=True)

datagen_test = ImageDataGenerator()

training_set = datagen_train.flow_from_directory('images/train',
                                                 target_size=(46,46),
                                                 batch_size=128,
                                                 color_mode = 'grayscale',
                                                 class_mode='categorical',
                                                 shuffle = True)

test_set = datagen_test.flow_from_directory('images/validation',
                                            target_size=(46,46),
                                            batch_size=128,
                                            color_mode = 'grayscale',
                                            class_mode='categorical',
                                            shuffle = True)

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout

model=Sequential()

model.add(Convolution2D(64,3,3,input_shape=(46,46,1),activation='relu'))    
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Convolution2D(128,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Convolution2D(256,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))


model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())


model.add(Dense(7,activation='softmax'))

opt = keras.optimizers.RMSprop(learning_rate=0.0001)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint("model_weights.h5",monitor = 'val_accuracy',save_weights_only = True, mode = 'max',verbose = 1)

reduce_lr =keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',factor = 0.1, patience = 2, min_lr = 0.00001,model = 'auto')

callbacks  = [PlotLossesCallback(), checkpoint, reduce_lr]

history = model.fit_generator(training_set,
                    steps_per_epoch=28821//128,
                    epochs=70,
                    validation_data=test_set,
                    validation_steps=7066//128,
                    callbacks= callbacks)


model.save('face_expression_model.h5')

###########     REPRESENTING THE MODEL AS A JSON STRING     #############
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
    
    

