import  pandas as pd
import numpy as np
#import sklearn.metrics as sspp

train = 'LMSalgtrain.csv'
test = 'LMSalgtest.csv'


def readInData(filename):
    df = pd.read_csv(filename)
    rows, columns = df.shape

    df.insert(0, 'Intercept', 1)

    X = df.iloc[:, 0:columns - 1]
    Y = df.iloc[:, columns - 1:columns]
    #print Y
    Z = df.iloc[:,-1]
    X = np.matrix(X.values)
    Y = np.matrix(Y.values)
    return X,Y,Z

O=readInData(train)
#print type(O)
X=O[0]
Y=O[1]
Zvalue=O[2]

rs, cols = X.shape
beta = np.zeros(shape=(1,cols))


def OLSData(X, Y, beta):
    alpha = 0.0001
    iters = 10000
    m = Y.size
    for i in  range(iters):
        h1 = np.matmul(X,beta.T)
        h1size = beta.T.size
        for j in range(h1size):
            var = X[:,j]
            var.shape = (m,1)
            MSE1 =  np.matmul((h1-Y).T, var)
            beta[0][j]= beta[0][j] - ((alpha/m) * MSE1.sum())
    return beta

def computeError(beta,X,Y):
    m = Y.size
    h1 = X.dot(beta.T)
    mse = np.matmul((h1 - Y).T, (h1 - Y))
    div = (1.0 / 2) * (1.0 / m)
    total = div * mse.sum()
    return 2 * total

beta =OLSData(X, Y, beta)
MSEtrain = computeError(beta,X,Y)

Otest=readInData(test)
Xtest=Otest[0]
Ytest=Otest[1]
Ztestvalue=Otest[2]
MSEtest = computeError(beta, Xtest, Ytest)

print('betas ' , beta)
print('MSE  is' , MSEtrain)
print('MSE test is' , MSEtest)

def calcWeights(X,Y,beta,MSE,epoch):
    eta = 0.00001
    mse=MSE;
    i=0
    #while (i <epoch):
    while (mse > 1 and (i < epoch)):
        i= i+1
        for k in range(X.shape[0]):
            h=np.matmul(X[k],beta.T)
            e = Y.item(k,0)-h
            bsize = beta.T.shape[0]
            for b in range(bsize):
                beta[0,b]= beta.item(0,b) + (eta * np.sum(e[0][0]) * X.item(k,b))

        mse = computeError(beta,X,Y)

    return  beta,mse

def calcWeightsO(X, Y, beta,MSE,epoch):
    eta = 0.00001
    mse = MSE;
    i=0
    while (mse > 1.0565 and i<epoch):
         i= i +1
         for k in range(X.shape[0]):
            h = np.matmul(X[k], beta.T)
            e = Y.item(k, 0) - h
            bsize = beta.T.shape[0]
            for b in range(bsize):
                 beta[0, b] = beta.item(0, b) + (eta * np.sum(e[0][0]) * X.item(k, b))
                 mse = computeError(beta, X, Y)
    return beta, mse

def calcWeightsMB(X, Y, beta,MSE,epoch):
    eta = 0.00001
    mse = MSE;
    batch = 150
    i=0
    while (mse > 1.0565 and i<epoch):
        i = i +1
        for k in range(X.shape[0]):
           batch = batch + 1
           h = np.matmul(X[k], beta.T)
           e = Y.item(k, 0) - h
           bsize = beta.T.shape[0]
           for b in range(bsize):
               beta[0, b] = beta.item(0, b) + (eta * np.sum(e[0][0]) * X.item(k, b))
           if(batch%150==0):
               mse = computeError(beta, X, Y)
               if(mse < 1.0565 ):
                 break;
    return beta, mse

epoch = 50
beta1=calcWeights(X,Y,beta,MSEtrain,epoch)
#print ('updated beats are ' , beta1[0])
#print ('updated mse is ' , beta1[1])
mse = computeError(beta1[0],Xtest,Ytest)
#print ('updated test mse is ' , mse)


beta1=calcWeightsO(X,Y,beta,MSEtrain,epoch)
#print ('updated online beats are ' , beta1[0])
#print ('updated online mse is ' , beta1[1])
mse = computeError(beta1[0],Xtest,Ytest)
#print ('updated online test mse is ' , mse)

beta1=calcWeightsMB(X,Y,beta,MSEtrain,epoch)
#print ('updated MB beats are ' , beta1[0])
#print ('updated MB mse is ' , beta1[1])
mse = computeError(beta1[0],Xtest,Ytest)
#print ('updated online test mse is ' , mse)

def calcY(X,beta):
    h1 = np.matmul(X, beta.T)
    return h1

Yhat = calcY(X,beta1[0])

predict = []
for i in range(len(Yhat)):
    if Yhat[i, 0] > 1:
        predict.append(1)
    else:
        predict.append(0)

def confusionMatrix(prediction, groundTruth):
    g1= groundTruth
    g2 = pd.Series(g1)
    conf = pd.crosstab(g2, prediction,rownames=['True'], colnames=['Predicted'], margins=False)
    return conf


groundTruth = pd.Series(Zvalue)
prediction = pd.Series(predict)

conf = confusionMatrix(prediction,Zvalue)
print conf


r , c = conf.shape
sum=0
check=0
for i in range(r):
    for j in range(c):
        if i==j:
            check = check + conf[i][j]
        sum = sum + conf[i][j]

accuracy = (check/float(sum))*100

acc = accuracy
print 'Accuracy = '+str(acc)+'%'

#test data
Yhattest = calcY(Xtest,beta1[0])

predicttest = []
for i in range(len(Yhattest)):
    if Yhattest[i, 0] > 1:
        predicttest.append(1)
    else:
        predicttest.append(0)

groundTruth = pd.Series(Ztestvalue)
prediction = pd.Series(predicttest)

conf = confusionMatrix(prediction,groundTruth)
print conf


r , c = conf.shape
sum=0
check=0
for i in range(r):
    for j in range(c):
        if i==j:
            check = check + conf[i][j]
        sum = sum + conf[i][j]

accuracy = (check/float(sum))*100

acc = accuracy
print 'Accuracy test = '+str(acc)+'%'


