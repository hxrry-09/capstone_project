from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import cv2
import os

#img = image.load_img(r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_train\asl_alphabet_train\A\A992.jpg")

img = cv2.imread(r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_train\asl_alphabet_train\A\A1.jpg")

##plt.imshow(img)
##plt.show()

print(cv2.imread(r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_train\asl_alphabet_train\A\A1.jpg"))
print(cv2.imread(r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_train\asl_alphabet_train\A\A1.jpg").shape)

train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory(r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_train\asl_alphabet_train"
                                          ,target_size=(200,200)
                                          ,batch_size=3001
                                          ,class_mode='categorical'
                                          )

validation_dataset = train.flow_from_directory(r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_validation\asl_alphabet_train"
                                          ,target_size=(200,200)
                                          ,batch_size=3001
                                          ,class_mode='categorical'
                                          )

print(train_dataset.classes)


model = tf.keras.models.Sequential([keras.layers.Conv2D(16,(3,3)) , activation = "relu"input_shape = (200,200,3),tf.keras.Maxpool2D(2,2) 
                                    
                                    keras.layers.Conv2D(32,(3,3)), activation ="relu",input_shape = (200,200,3),tf.keras.Maxpool2D(2,2) 
                                   
                                    keras.layers.Conv2D(64,(3,3)), activation ="relu",input_shape = (200,200,3),tf.keras.Maxpool2D(2,2) 
                                    
                                    keras.layers.Flatten()
                                    keras.layers.Dense(512,activation = 'relu')
                                    
                                    keras.layers.Dense(1,activation='softmax')
                                    ])

