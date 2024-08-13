import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
datadir=tf.keras.utils.get_file('flower_photos', origin=url, untar=True, cache_dir='.')

import pathlib

datadir=pathlib.Path(datadir)
count=len(list(datadir.glob('*/*.jpg')))


roses=list(datadir.glob('roses/*'))

import PIL
PIL.Image.open(str(roses[25]))


flowers_images_dict={
    'roses': list(datadir.glob("roses/*")),
    'tulips': list(datadir.glob("tulips/*")),
    'dandelion': list(datadir.glob("dandelion/*")),
    'daisy': list(datadir.glob("daisy/*")),
    'sunflowers': list(datadir.glob("sunflowers/*"))
}

flower_labels_dict={
    'roses': 0,
    'tulips': 1,
    'dandelion': 2,
    'daisy': 3,
    'sunflowers': 4
}
import cv2
images=cv2.imread(str(flowers_images_dict['roses'][0]))
images=cv2.resize(images,(180,180))


x, y = [], []
for flower_name, images in flowers_images_dict.items():
    for img in images:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (180, 180))
        img = img / 255.0
        x.append(img)
        y.append(flower_labels_dict[flower_name])
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0)


xtrain=xtrain/255
xtest=xtest/255
classes=5

cnn=tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3, padding='same', activation='relu', input_shape=(180,180,3)))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3, padding='same',activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=classes, activation='softmax'))

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(xtrain, ytrain, epochs=30, validation_data=(xtest, ytest))