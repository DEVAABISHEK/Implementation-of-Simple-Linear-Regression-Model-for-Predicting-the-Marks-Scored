# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries
Use Pandas, NumPy, Matplotlib, and Scikit-learn.

2.Load the dataset
Read the dataset (student_scores.csv) into a dataframe.

3.Separate variables
Independent variable: X (Hours studied)
Dependent variable: Y (Marks scored)

4.Split the dataset into training and testing sets:
X_train, X_test, Y_train, Y_test
	​
5.Train the Linear Regression model using the equation: y = b0 + b1x
Where:
b1 (slope) = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)²
b0 (intercept) = ȳ − b1x̄

6.Make predictions using: Y_pred = b0 + b1X_test

7. Evaluate the model using:
MSE = (1/n) Σ(Y_test − Y_pred)²
MAE = (1/n) Σ|Y_test − Y_pred|
RMSE = √MSE

## Program:
~~~
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DEVA ABISHEK P
RegisterNumber:  212223110008

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
~~~

## Output:
## Dataset:
![image](https://github.com/user-attachments/assets/ff403900-a747-4eb1-a94e-9ba845ac490c)
## Y predicted:
![image](https://github.com/user-attachments/assets/bf60c08f-ab6d-47d5-884c-60ee3d8cc1bd)
## Training set:
![image](https://github.com/user-attachments/assets/f4f647c6-5c5d-4ff0-8006-130ce27dc72c)
## Values of MSE,MAE,RMSE:
![image](https://github.com/user-attachments/assets/5f191082-4458-40bf-9a62-03c18a34f175)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
