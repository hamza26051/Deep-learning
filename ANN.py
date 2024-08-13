import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from sklearn.compose import ColumnTransformer
data=pd.read_csv('/content/Churn_Modelling.csv')
x=data.drop(columns=['RowNumber','CustomerId','Surname','Exited'])
y=data['Exited']

categorical_cols=['Geography','Gender']
ct=ColumnTransformer(transformers=[('encode', OneHotEncoder(), categorical_cols)
                                   ], remainder='passthrough')
x=np.array(ct.fit_transform(x))
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
ss=StandardScaler()
xtrain=ss.fit_transform(xtrain)
xtest=ss.transform(xtest)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(xtrain, ytrain, batch_size=32, epochs=100)



sample_input = np.array([['France', 'Female', 600, 40, 3, 60000, 2, 1, 1, 50000]])

sample_df = pd.DataFrame(sample_input, columns=[
    'Geography', 'Gender', 'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
])

sample_df_transformed = ct.transform(sample_df)
sample_df_transformed = ss.transform(sample_df_transformed)

prediction = ann.predict(sample_df_transformed)
print(prediction)