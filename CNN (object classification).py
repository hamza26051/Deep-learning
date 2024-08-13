import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
trainingprocess=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
trainingdata= trainingprocess.flow_from_directory("/content/drive/MyDrive/training_set",
                                                  target_size=(32,32),
                                                  batch_size=32,
                                                  class_mode='binary')
testingprocess=ImageDataGenerator(rescale=1./255)
testdata= testingprocess.flow_from_directory("/content/drive/MyDrive/test_set",
                                                  target_size=(32,32),
                                                  batch_size=32,
                                                  class_mode='binary')

cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[32,32,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='softmax'))

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x=trainingdata, validation_data=testdata, epochs=15)

import numpy as np
from keras.preprocessing import image
imager=image.load_img("/catpic.jpg", target_size=(32,32))
imager=image.img_to_array(imager)
imager=np.expand_dims(imager, axis=0)
result=cnn.predict(imager)
if result[0][0]==1:
    prediction="dog"
else:
    prediction="cat"
print(prediction)