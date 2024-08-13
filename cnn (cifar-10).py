import tensorflow as tf
from tensorflow.keras.datasets import cifar10


(xtrain, ytrain), (xtest, ytest)=cifar10.load_data()

trainingpro=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
testpro=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

traindata=trainingpro.flow(xtrain,ytrain,  batch_size=32)
testdata=testpro.flow(xtest,ytest, batch_size=32)

cnn=tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu', input_shape=[32,32,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
batch_size=32
cnn.fit(x=traindata,validation_data=testdata,steps_per_epoch=len(xtrain)//batch_size,epochs=60, validation_steps=len(xtest)//batch_size)