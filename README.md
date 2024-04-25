# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: N Preethika
RegisterNumber:  212223040130
*/
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

Placement data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/1e431f79-d203-4712-b49b-eacb614ec319)

Salary

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/e664eab2-e3ad-4131-8981-7637ae9a1803)

Status data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/2970d84f-c923-4dde-9461-0149af0d1c33)

Duplicate data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/2e958128-b0cc-4ac9-bb04-e039bc0da731)

Data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/815b6727-ce50-4d80-9eea-41bfd069a6f4)

X Data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/205fe7bf-7ba5-4919-9aa2-090661a24e71)

Y Status

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/12b195fa-16cf-4e37-87b0-6d1b961a01bc)

Accuracy

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/b4e12490-52e7-4d4d-b0a2-b59dd24aec75)

Confusion Data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/1c96aa97-2ede-417b-aebf-247b8c1f68d2)

Classification data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/6429f203-74bb-494f-aaa4-0ba308d96f79)

Predicated Data

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/f2794852-a4d8-47e9-823d-62acd78744d3)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

