import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('trainPredictors.csv')

rows , columns = df.shape

for i in range(1,columns):
    mean1 = df.mean()[i]
    std1 =  df.std()[i]

    for j in range(rows):
        #print df.iat[j,i]
        df.iloc[j,i] = (df.iloc[j,i] - mean1)/std1
   # stdev[i]= df.ix[:,i]

df.insert(0,'Intercept',1)

df1 = pd.read_csv('testPredictors.csv')

X=df.iloc[:,0:columns]
Y=df.iloc[:,columns:columns+1]
rs,cols = X.shape
X = np.matrix(X.values)
Y = np.matrix(Y.values)

beta = np.zeros(shape=(1,cols))

alpha = 0.01
iters = 1000


def costFunction(X, Y, beta):
    m = Y.size

    h1 = X.dot(beta.T)

    MSE =np.matmul((h1-Y).T,(h1-Y))
    div = (1.0/2) *(1.0/m)
    total =   div * MSE.sum()
    return total


def gradientDescent(X, Y, beta, alpha, iters):
    m = Y.size
    cost = []
    for i in  range(iters):
        #h1 = X.dot(beta.T)
        h1 = np.matmul(X,beta.T)
        #print 'h1 is ' , h1.shape
        h1size = beta.T.size
        for j in range(h1size):
            #print 'j is' , j
            var = X[:,j]
            var.shape = (m,1)
            #print 'var size is 2', var.shape
            MSE1 =  np.matmul((h1-Y).T, var)
            beta[0][j]= beta[0][j] - ((alpha/m) * MSE1.sum())
            #print beta[0][j]
        cost.append(costFunction(X,Y,beta))
    return beta, cost

result =gradientDescent(X, Y, beta, alpha, iters)
#print result


def gradientDescentRidge(X, Y, beta, alpha, iters, ridgeLambda):
    m = Y.size
    cost = []
    for i in  range(iters):
        h1 = X.dot(beta.T)
        #print 'h1 is ' , h1.shape
        h1size = beta.T.size
        for j in range(h1size):
            #print 'j is' , j
            var = X[:,j]
            #print 'var size is' , var.shape
            var.shape = (m,1)
            #print 'var size is 2', var.shape
            #MSE1 =  np.matmul((h1-Y).T, var)
            MSE1 = (h1-Y).T.dot(var)
            #print 'h1-y shape ' , (h1-Y).T.shape
            #print 'MSE SHape 1', MSE1.shape
            beta[0][j]= beta[0][j] - (alpha/m) * (MSE1.sum() + (ridgeLambda * beta[0][j]))
        cost.append(costFunction(X,Y,beta))
           #beta[j] = beta[j] - (alpha/X * np.alen(X)
    return beta, cost


print result[0]
beta = np.zeros(shape=(1,columns))
alpha = 0.01
itersreg = 1000
ridgeLambda = 0.05
regResult = gradientDescentRidge(X, Y, beta, alpha, itersreg, ridgeLambda)
print regResult[0]


def MSE(beta):
    '''
    Compute and return the MSE.
    '''
    mse = 2 * costFunction(X,Y,beta)
    return mse

beta = regResult[0]
print 'MSE is ', MSE(beta)
