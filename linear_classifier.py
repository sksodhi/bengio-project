#
# Linear classifier - Multiclass with Kesler's construction
#

# coding: utf-8

# In[301]:

import read_dataset
import numpy as np
import sys
sys.path.append('..')


# In[302]:

def splitdata(X,T,trainpct):
    #X is w/o x0. T is a list of letter strings, trainpct is % of data to be the training set
    T_let = np.asarray(T, dtype=np.unicode_).reshape(len(T),) #change to unicode array
    
    permuted_indices = np.random.permutation(len(X)); #divide data into two subset train and test
    split = int(trainpct/100.0*len(X))
    
    train_i = permuted_indices[:split]
    test_i = permuted_indices[split:]
    return X[train_i],X[test_i],T_let[train_i],T_let[test_i]


# In[303]:

def linearRegressionTraining(X,T): 
    #X is w/o x0. T is an array of letter strings
    
    T_KS= np.full((len(T),ord('Z')- ord('A')+1), -1, dtype=np.float) #Kessler construction
    for i,t in enumerate(T):
        T_KS[i][ord(t)-ord('A')]=1.0
    
    #Augmented X
    X0 = np.full((len(X),1), 1, dtype=np.float)
    X_a = np.hstack((X0,X))
    
    #Calculate the pseudo-inverse Matrix of X, by using numpy
    X_pinv = np.linalg.pinv(X_a)
    
    W = np.dot(X_pinv,T_KS)
    return W


# In[304]:

def applyLinearClassifier(X,W):
    X0 = np.full((len(X),1), 1, dtype=np.float)
    X_a = np.hstack((X0,X))
    TP = np.dot(X_a,W)
    resultlabel_i = np.argmax(TP, axis=1) #index of maximum on horizental axis
    resultlabel_u=resultlabel_i+ord('A')
    resultlabel = [chr(l) for l in resultlabel_u]
    return resultlabel,resultlabel_i


# In[305]:

def computeConfusionMatrix(resultlabel_i,T): 
    #T is ground truth letter array. resultlabel_i is classified label index (columnwise)
    num=ord('Z')- ord('A')+1
    Mconf = np.full((num,num), 0, dtype=np.int)
    for truth in range(num):
        for predict in range(num):
            mark = np.full((len(resultlabel_i),), 0, dtype=np.int)
            mark[((resultlabel_i==predict)&(T==chr(truth+ord('A'))))]=1
            Mconf[truth][predict]=mark.sum()
            
    ppv_denom = np.sum(Mconf,axis=0) #it's a vector that sums each column
    ppv_top = np.full((num,), 0, dtype=np.int)
    for i in range(num):
        ppv_top[i]= Mconf[i][i]

    ppv = ppv_top*100.0/ppv_denom; #a vector for all classes

    ppvmax_ind = np.argmax(ppv)
    ppvmin_ind = np.argmin(ppv)
    ppvmax=chr(ppvmax_ind+ord('A'))
    ppvmin=chr(ppvmin_ind+ord('A'))
    return Mconf,ppv,ppvmax,ppv[ppvmax_ind],ppvmin,ppv[ppvmin_ind]

#
# Linear Classifier
#
class LinearClassifier():
    def __init__(self, X, T, debug):
        self._X = X
        self._T = T
        self._debug = debug

    #
    # Build Classifier
    #
    def build(self):
        print ("LinearClassifier: building classifier")
        self._W = linearRegressionTraining(self._X, self._T)

    #
    # Apply Classifier
    #
    def classify(self, Xtest):
        print ("LinearClassifier: applying classifier")
        resultlabel, resultlabel_i=applyLinearClassifier(Xtest,self._W)

        return resultlabel


"""

# In[306]:

X,T = read_dataset.read_letter_recognition_dataset('./datasets/letter-recognition/letter-recognition.data')

# In[307]:

X_train,X_test,T_train,T_test = splitdata(X,T,75)


# In[308]:

W = linearRegressionTraining(X_train,T_train)


# In[309]:

resultlabel,resultlabel_i=applyLinearClassifier(X_test,W)


# In[310]:

Mconf,ppv,ppvmax_label,ppvmax,ppvmin_label,ppvmin=computeConfusionMatrix(resultlabel_i,T_test)


# In[311]:

print(ppvmax_label,ppvmax,ppvmin_label,ppvmin,ppv)

"""

