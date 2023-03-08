import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

#Define the image size classes
img_size = (64, 64)
num_classes = 29

#paths 
train_path = r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_train\asl_alphabet_train"
test_path = r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_test\asl_alphabet_test"
val_dir = r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_validation\asl_alphabet_validation"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = img_size,
    batch_size = 29,
    class_mode ='categorical')

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = img_size,
    batch_size = 29,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = img_size,
    batch_size = 29,
    class_mode = 'categorical')

"""""
print("Number of training samples:", train_generator.n)
print("Number of validation samples:", val_generator.n)
print("Number of test samples:", test_generator.n)
"""""

#model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
  
# model training

model.fit(train_generator,
          steps_per_epoch=train_generator.n // 29,
          epochs = 5,
          validation_data=test_generator,
          validation_steps=test_generator.n // 29)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.n // 29)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


model.save(r"C:\Users\harry\Downloads\Cap_stone\python_files\asl_model.keras")
