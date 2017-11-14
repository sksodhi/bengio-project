
# coding: utf-8

# In[500]:


import numpy as np

def bin_size(N):
    '''
    This function calculates bin size
    
    **************
    Input: N - number of samples (int)
    
    Output: B - bin size (int)
    '''
    return int(2*np.power(int(N),1/3.0))
    

def bin_indexing(X,B,xmin,xmax):
    '''
    This function calculates bin index
    
    **************
    Input: X - input value (matrix,array)
           B - bin size (int)
           xmin, xmax - maximum and minimum value of input (numeric)
    
    Output: Bin index
    '''
    return np.round((B-1)*(X-xmin)/((1.0)*(xmax-xmin)))

def get_labels(T):
    '''
    This function computes set of class labels
    
    **************
    Input: T - class labels (matrix,array)
           
    Output: set of class labels
    '''
    return set(T)

def hist_preprocess(B,no_features,labels):
    '''
    This function creates histogram dictionary
    
    **************
    Input: B - bin size (int)
           no_features - number of features (int)
           labels - class labels (anytype)
    
    Output: H - histogram dictionary (dictionary)
    '''
    H = {str(key):{} for key in labels}
    return H

def min_max(X,no_features):
    '''
    This function calculates min and max array for each feature from the given input
    
    **************
    Input: X - input (matrix)
           no_features - number of features (int)
    
    Output: feature_min - array of min value for each feature (array)
            feature_max - array of max value for each feature (array)
    '''
    x = np.transpose(X)
    feature_min = np.empty(no_features)
    feature_max = np.empty(no_features)
    for i in xrange(0,no_features):
        feature_min[i] = np.min(x[i])
        feature_max[i] = np.max(x[i])
    return feature_min, feature_max

def histogram_train(X,T,H,B,feature_min,feature_max):
    '''
    This function trains histogram classfier
    
    **************
    Input: X - input (matrix)
           T - class labels (matrix or array)
           H - histogram dictionary
           feature_min - array of min value for each feature (array)
           feature_max - array of max value for each feature (array)
    
    Output: void
    '''
    for index,x in enumerate(X):
        bin_idx = str(bin_indexing(x,B,feature_min,feature_max))
        H[str(T[index])].setdefault(bin_idx, 0)
        H[str(T[index])][bin_idx]+=1
        
def histogram_class(queries,labels,B,feature_min,feature_max):
    '''
    This function classifies label based on the given query
    
    **************
    Input: queries - query input (1D array)
           labels - class labels (1D array)
           H - histogram dictionary
           feature_min - array of min value for each feature
           feature_max - array of max value for each feature
    
    Output: result - dictionary with 
                labels => key
                number in histogram => value
            result_label - class label (one value)
    '''
    result = {}
    bin_idx = str(bin_indexing(queries,B,feature_min,feature_max))
    for label in labels:
        if bin_idx in H[label]:
            result[label] = H[label][bin_idx]
        else:
            result[label] = 0
    result_label = [k for k, v in result.iteritems() if v == np.max(result.values())]
    return result,result_label
    


    


# In[501]:


import numpy as np

#
# This function reads letter recognition dataset
# Dataset location: http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
#
def read_letter_recognition_dataset(data_file_name):
    num_rows=20000
    num_cols=16
    X = np.empty((num_rows, num_cols))
    T = []
    r = 0
    data_file = open(data_file_name, 'r')
    for line in data_file:
        c = 0
        for field in line.strip().split(','):
            if c == 0:
                T.append(field)
            else:
                x_col = c - 1
                X[r][x_col] = field
            c += 1
        r += 1
            #print (field)

    return X,T


# In[502]:


X,T = read_letter_recognition_dataset('./datasets/letter-recognition/letter-recognition.data')


# In[504]:


### Set parameters

x_size  = np.shape(X); # dimensions of input array
labels = get_labels(T); # get labels
no_features = x_size[1] # get number of features
B = bin_size(no_features) # get bin number
H = hist_preprocess(B,no_features,labels) # histogram dictionary
feature_min, feature_max = min_max(X,no_features) # min and min for each feature


# In[505]:


### Train histogram
histogram_train(X,T,H,B,feature_min,feature_max)


# In[507]:


### Test histogram
idx = 11
queries = X[idx] # feature vector
label_q = T[idx] # label of query
hist_dic, hist_label = histogram_class(queries,labels,B,feature_min,feature_max)
print ("Histogram:",hist_label,"True label:",label_q)

