#Simple Linear Regression with one feature

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SimpleLinearRegression:
    
    def fit(self, X, Y, regularization):
        self.X = X
        self.Y = Y
        theta = np.zeros(2)
        alpha = 0.01
        num_iter = 1500
        train = Train(self.X, self.Y)
        self.theta, self.cost_history = train.gradient_descent(theta, alpha, num_iter, regularization)
    
    def predict(self, X):
        test = Test(X)
        self.Yhat = test.get_predictions(self.theta)
        print(self.theta)
        return self.Yhat
    
    def cost(self):
        train = Train(self.X, self.Y)
        return train.cost_function(self.theta)
    
    def plot_cost(self):
        plt.grid(True)
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
        
    #Plot data
    def plotData(self,X, Y, Yhat):
        plt.grid(True)
        plt.scatter(X, Y, marker='x', color = 'red')
        plt.plot(X, Yhat)
        plt.xlim(4, 24)
        plt.ylim(ymin=0)
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.show()
        
    def score(self, Y_test, Y_pred):
        sserror = np.sum((Y_test - Y_pred)**2)
        sstot = np.sum((Y_test - np.mean(Y_test))**2)
        return 1 - (sserror/sstot)
    
class Train:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def gradient_descent(self, theta, alpha, num_iter, regularization):
        new_theta = np.zeros(2)
        new_theta[0] = theta[0]
        new_theta[1] = theta[1]
        cost_history = []
        l = 1e-3
        for j in range(num_iter):
            for i in range(theta.size):
                f = np.sum((np.dot(self.X, theta) - self.Y) * self.X[:, i])
                if regularization == 'l1':
                    r = l*np.sign(new_theta[i])
                else:
                    r = l*(new_theta[i]**2)
                new_theta[i] = new_theta[i] - (alpha/self.Y.size) * (f + r)
            for i in range(theta.size):
                theta[i] = new_theta[i]
            if j%10 == 0:
                cost = self.cost_function(theta)
                cost_history.append(cost)
        return theta, cost_history
    
    def cost_function(self, theta):
        return np.sum((np.dot(self.X, theta) - self.Y)**2)/(2*self.Y.size)
    
class Test:
    def __init__(self, X):
        self.X = X
    
    def get_predictions(self, theta):
        return np.dot(self.X, theta)

df = pd.read_csv("Simple_LR.txt", sep=',', header=None)
df.insert(0, 'B', np.ones(df.shape[0]))

#Data Preprocessing
X = np.array(df.iloc[:, :-1])
Y = np.array(df.iloc[:, -1])

#model fit
model = SimpleLinearRegression()
model.fit(X, Y, regularization='l1')
print("Training Cost: ", model.cost())
Y_pred = model.predict(X)
print("Training Score: ", model.score(Y, Y_pred))

#Plot
model.plotData(X[:, 1], Y, Y_pred)
model.plot_cost()

#Predict
predict1 = model.predict([[1, 3.5]])
print('For population = 35,000, we predict a profit of ' + str(predict1*10000))
predict2 = model.predict([[1, 7]])
print('For population = 70,000, we predict a profit of ' + str(predict2*10000))

test = Test(X)
yh1 = test.get_predictions([-2, 0.5])
yh2 = test.get_predictions([-3.63023346, 1.16635641])

plt.grid(True)
plt.scatter(X[:, 1], Y, marker='x', color = 'red')
plt.plot(X[:, 1], yh1, 'g', X[:, 1], yh2, 'b')
plt.xlim(4, 24)
plt.ylim(ymin=0)
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()