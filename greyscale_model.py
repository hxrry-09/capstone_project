import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

#Define the image size classes
img_size = (200, 200)
num_classes = 29

train_path = r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_train\asl_alphabet_train"
test_path = r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_test\asl_alphabet_test"
val_dir = r"C:\Users\harry\Downloads\Cap_stone\asl_alphabet_validation"

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = img_size,
    batch_size = 29,
    class_mode ='categorical',
    color_mode ='grayscale'
    )

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = img_size,
    batch_size = 29,
    class_mode='categorical',
    color_mode='grayscale'
    )

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = img_size,
    batch_size = 29,
    class_mode = 'categorical',
    color_mode = 'grayscale'
    )

model = Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(256, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(29, activation='softmax')
])

opt = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy']
            )

model.fit(train_generator,
          steps_per_epoch=train_generator.n // 29,
          epochs = 10,
          validation_data=test_generator,
          validation_steps=test_generator.n // 29)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.n // 29)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)