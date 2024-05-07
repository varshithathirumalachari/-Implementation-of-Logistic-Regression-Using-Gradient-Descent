# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Start by initializing the parameters (weights) theta with random values or zeros.
2. Compute Sigmoid Function: Define the sigmoid function that maps any real-valued number to a value between 0 and 1.
3. Compute Loss Function: Define the loss function, which measures the error between the predicted output and the actual output.
4. Gradient Descent Optimization: Implement the gradient descent algorithm to minimize the loss function. In each iteration, compute the gradient of the loss function with respect to the parameters (theta), and update the parameters in the opposite direction of the gradient to minimize the loss.
5. Iterate Until Convergence: Repeat the gradient descent steps for a predefined number of iterations or until convergence criteria are met. Convergence can be determined when the change in the loss function between iterations becomes very small or when the parameters (theta) stop changing significantly.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VARSHITHA A T
RegisterNumber: 212221040176
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:
### Dataset
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/29607282-3e0c-497c-920b-3bab31eeac50)


### Data Types
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/7aed4d29-2296-44e2-b800-f8e679f5c405)


### New Dataset
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/394ea2e1-bea0-4a8f-93c7-f90a276a4f82)


### Y values
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/b91e0ebb-5f1b-46b3-adf4-34738d64290a)

### Accuracy
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/df699582-fba4-4828-886d-64f5b7c64589)



### Y pred
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/edb0ddec-db8f-4dcb-95ec-ec4096a0f47f)


### New Y 
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/f4878ce8-846f-4b53-bbee-af3f2d775d82)

![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/e9157f6c-73ad-457e-a647-106f96040c22)
![image](https://github.com/varshithathirumalachari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131793193/dc759d66-6947-439f-8286-6392379ccbb9)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

