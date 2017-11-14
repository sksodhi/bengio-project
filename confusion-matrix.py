
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 


# In[2]:

#Confusion Matrix for Classifier
confusionmatrix=metrics.confusion_matrix(true_class, predict_class) 
print(confusionmatrix)
plt.title('confusion matrix')
plt.imshow(confusionmatrix, cmap='binary', interpolation='None')
plt.show()


# In[3]:

"""
#Traditional approach
def Confusion_matrix(t_c,p_c):
    B=np.count_nonzero(np.unique(t_c))+1;
    cm = np.zeros((B,B)).astype('int32');
    #print(cm);
    
    for i,j in zip(t_c,p_c):
        cm[i][j] += 1;
    
    return cm;

"""


# In[4]:

#PPV for classifier
confusionmatrixT=confusionmatrix.T
ppvm=list();
for i,item in enumerate(confusionmatrixT):
    ppvm.append(confusionmatrixT[i][i]/np.sum(item));    
    
print(ppvm)
highestPPV=np.amax(ppvm)*100
highestPPVindex=np.argmax(ppvm)
lowestPPV=np.amin(ppvm)*100
lowestPPVindex=np.argmin(ppvm)

print(highestPPV)
print(highestPPVindex)
print(lowestPPV)
print(lowestPPVindex)


# In[5]:

#Accuracy
accuracy=metrics.accuracy_score(true_class, predict_class)
print(accuracy)

#Sensitivity
sensitivity=metrics.recall_score(true_class, predict_class) #Only for binary classifier
print(sensitivity)


# In[6]:

#Receiver Operating Characteristic (ROC) Curve 
#Only for binary Classifier 
fpr, tpr, thresholds = metrics.roc_curve(true_class, predict_class)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[7]:

#AUC, auc=1 perfect classifier, auc>0.8 satisfactory classifier, auc=0.5 classifier performing no better than random decisions
#Only for binary Classifier 
auc=metrics.roc_auc_score(true_class, predict_class)
print(auc)

