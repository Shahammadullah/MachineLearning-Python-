
# coding: utf-8


from __future__ import division
import numpy as np
import pandas as pd
#import scipy.optimize as opt



# Load the LogisticData.csv for this assignment and check if it has loaded successfully. In this excercise, we will use the same dataset for training and testing. After training the model on the best beta, we will see how well does our model perform.
df = pd.read_csv('LogisticData.csv')

rows,columns = df.shape
#print rows,columns
for i in range(columns-1):
    #a = []
    #a = df.iloc[:,i]
    mean1 = df.mean()[i]
    std1 =  df.std()[i]
    #print  mean1
    #print std1
    for j in range(rows):
        #print df.iat[j,i]
        df.iloc[j,i] = (df.iloc[j,i] - mean1)/std1
# Next, preprocess the data. X should contain only the predictors, Y shuould be the reponse variable and beta should be a vector with length = the number of features and value = 0.
X = df.iloc[:,0:columns-1]
Y = df.iloc[:,columns-1:columns]

dfn = pd.DataFrame()
dfn['Result'] = Y
#print X
#print 'y is '
#print Y
groundTruth = dfn['Result']
#Normalize only if you are using gradient descent. Apply standard deviation for normalization.

X = np.matrix(X.values)  
Y = np.matrix(Y.values)
beta = np.zeros(shape=(1,columns-1))


# Define a sigmoid function and return the value tht has been calculated for z
def sigmoid(z):
    '''
    Here sigmoid value of 'z' needs to be returned.
    '''
    x = 1 + np.exp(-z)
    sig = 1/x
    return sig


# Define the cost function for Logistic Regression. Remember to calculate the sigmoid values as well.
def costFunction(beta, X, Y):

    '''
    This function returns the value computed from the cost function.
    '''
    m = X.shape[0]
    #print 'm is ', m
    h1 = X.dot(beta.T)
    result = (-1.0/m)* (np.matmul((Y.T),np.log(sigmoid(h1))) + np.matmul((1-Y).T,np.log(1-sigmoid(h1)))).sum()
    # print h1
    #MSE = np.matmul((h1 - Y).T, (h1 - Y))
    #print 'MSE size ', MSE.sum()
    # MSE = (h1-Y).T * (h1-Y)
    #div = (1.0 / 2) * (1.0 / m)
    #total = div * MSE.sum()
    #print total
    #return total
    return result


# Define a gradient function that takes in beta, X and Y as parameters and returns the best betas and cost. 
def gradientDescent(X, Y, beta, alpha, iters):
    '''
    Compute the gradient descent function.
    Return the newly computed beta and the cost array.
    '''
    m = Y.size
    cost = []
    for i in range(iters):
        h1 = X.dot(beta.T)
        h2 = sigmoid(h1)
        # print 'h1 is ' , h1.shape
        h1size = beta.T.size
        for j in range(h1size):
            # print 'j is' , j
            var = X[:, j]
            # print 'var size is' , var.shape
            var.shape = (m, 1)
            # print 'var size is 2', var.shape
            # MSE1 =  np.matmul((h1-Y).T, var)
            MSE1 = (h2 - Y).T.dot(var)
            # print 'h1-y shape ' , (h1-Y).T.shape
            # print 'MSE SHape 1', MSE1.shape
            beta[0][j] = beta[0][j] - alpha * (1.0 / m) * MSE1.sum()
        cost.append(costFunction(beta,X,Y))
        # beta[j] = beta[j] - (alpha/X * np.alen(X)
    return beta, cost



# Try out multiple values of 'alpha' and 'iters' so that you get the optimum result.
#please try different values to see the results, but alpha=0.01 and iters=10000 are suggested.
alpha = 0.01
iters = 10000
result = gradientDescent(X, Y, beta, alpha, iters)
print result


# Now , only define the gradient function that we can use in the SciPy's optimize module to find the optimal betas. 
def gradient(beta, X, Y):
    '''
    This function returns the gradient calucated.
    '''
    #for i in range(parameters):
    #####
    #grad[i] =
    return grad


# Optimize the parameters given functions to compute the cost and the gradients. We can use SciPy's optimization to do the same thing.
# Define a variable result and complete the functions by adding the right parameters.

#the optimised betas are stored in the first index of the result variable
#result = opt.fmin_tnc(func= , x0= , fprime = , args= )


# Define a predict function that returns 1 if the probablity of the result from the sigmoid function is greater than 0.5, using the best betas and 0 otherwise.
def predict(beta, X): 
    '''
    This function returns a list of predictions calculated from the sigmoid using the best beta.
    '''
    h1 = X.dot(beta.T)
    h2 = sigmoid(h1)
    predict = []
    for i in range(len(h2)):
        if h2[i, 0] > 0.5:
            predict.append(1)
        else:
            predict.append(0)
    return predict


# Store the prediction in a list after calling the predict function with best betas and X.
bestBeta = np.matrix(result[0])
predictions = predict(bestBeta, X)


# Calculate the accuracy of your model. The function should take the prediction and groundTruth as inputs and return the 
# confusion matrix. The confusion matrix is of 'dataframe' type.
def confusionMatrix(prediction, groundTruth):
    '''
    Return the computed confusion matrix.
    '''
    return pd.crosstab(groundTruth, prediction, rownames=['True'], colnames=['Predicted'], margins=False)


# Call the confusionMatrix function and print the confusion matrix as well as the accuracy of the model.
#The final outputs that we need for this portion of the lab are conf and acc. Copy conf and acc in a .txt file.
#Please write a SHORT report and explain these results. Include the explanations for both logistic and linear regression
#in the same PDF file. 

groundTruth = pd.Series(groundTruth)
prediction = pd.Series(predictions)
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

accuracy = (check/sum)*100

acc = accuracy
print 'Accuracy = '+str(acc)+'%'

