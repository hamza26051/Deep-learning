import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

scaling = 1000
train_df['median_house_value'] /= scaling
test_df['median_house_value'] /= scaling

train_df = train_df.reindex(np.random.permutation(train_df.index))

threshold = 265000/scaling

trainmean = train_df.mean()
train_mean = train_df.mean()
train_std = train_df.std()
trainstd = train_df.std()

traindfnorm = (train_df - trainmean) / trainstd
testdfnorm = (test_df - trainmean) / trainstd

traindfnorm['median_house_value_ishigh'] = (train_df['median_house_value'] > threshold).astype(float)
testdfnorm['median_house_value_ishigh'] = (test_df['median_house_value'] > threshold).astype(float)

inpute = {
    'median_income': tf.keras.Input(shape=(1,)),
    'total_rooms': tf.keras.Input(shape=(1,)),
    'longitude': tf.keras.Input(shape=(1,)),
    'latitude': tf.keras.Input(shape=(1,)),
    'housing_median_age': tf.keras.Input(shape=(1,)),
    'total_bedrooms': tf.keras.Input(shape=(1,)),
    'population': tf.keras.Input(shape=(1,)),
    'households': tf.keras.Input(shape=(1,))
}

def buildmodel(input, learning, Metrics):
    concatenatedlayer = tf.keras.layers.Concatenate()(list(input.values()))
    x = tf.keras.layers.Dense(units=100, activation='sigmoid')(concatenatedlayer)
    x = tf.keras.layers.Dense(units=100, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(units=100, activation='sigmoid')(x)
    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=Metrics)
    return model

def trainmodel(model, df, epochs, labelname, batch, validation):
    features = {name: np.array(value) for name, value in df.items()}
    labels = np.array(features.pop(labelname))

    history = model.fit(features, labels, epochs=epochs, batch_size=batch, validation_split=validation, shuffle=True)
    histdf = pd.DataFrame(history.history)
    return history.epoch, histdf

learning = 0.01
validation = 0.2
batch = 32
epochs = 30
labelname = "median_house_value_ishigh"
classificationthreshold = 0.35

Metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classificationthreshold)]

mymodel = buildmodel(inpute, learning, Metrics)
epochs, hist = trainmodel(mymodel, traindfnorm, epochs, labelname, batch, validation)

testfeatures = {name: np.array(value) for name, value in testdfnorm.items()}
testlabels = np.array(testfeatures.pop(labelname))

mymodel.evaluate(testfeatures, testlabels, batch_size=batch)

predictions = mymodel.predict(testfeatures)

for i in range(len(testlabels)):
    print(f"actual value: {testlabels[i]} predicted value: {predictions[i][0]}")

