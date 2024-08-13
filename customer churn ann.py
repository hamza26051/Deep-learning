import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.compose import ColumnTransformer

data=pd.read_csv("churn.csv")
x=data.drop(columns=["customerID",'Churn'])
y=data['Churn']

print(data.info())

numerical_features=['SeniorCitizen','tenure','MonthlyCharges']
categorical_features=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','TotalCharges']


preprocess=ColumnTransformer(transformers=[
                                           ('encode', OneHotEncoder(),categorical_features )], remainder='passthrough')

x=preprocess.fit_transform(x)

le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

ss=StandardScaler(with_mean=False)
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

ann=tf.keras.Sequential()
ann.add(tf.keras.layers.Dense(units=20,activation='relu'))
ann.add(tf.keras.layers.Dense(units=20,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(x_train, y_train, epochs=100, batch_size=32)
ann.evaluate(x_test, y_test)
sample=np.array([['Male',0,'Yes', 'No', 30, 'No','No phone service','DSL',"Yes",'Yes','NO','NO','NO',"NO",'Month-to-month','No', 'Mailed check',40,400]])
sampledf=pd.DataFrame(sample,columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])
sampledftransformed=preprocess.transform(sampledf)
sampledftransformed=ss.transform(sampledftransformed)

prediction=ann.predict(sampledftransformed)
print(prediction)