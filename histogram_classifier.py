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

def histogram_train(X,T,B,H,feature_min,feature_max):
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
        
def histogram_class(queries,labels,B,H,feature_min,feature_max):
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
    result_label = []
    bin_idx = str(bin_indexing(queries,B,feature_min,feature_max))
    for label in labels:
        if bin_idx in H[label]:
            result[label] = H[label][bin_idx]
        else:
            result[label] = 0
    if(np.max(result.values())):
    	result_label = [k for k, v in result.iteritems() if v == np.max(result.values())]
    else:
    	import random
    	result_label = [random.choice(result.keys())]
    return result,result_label

#
# Histogram Classifier
#
class HistogramClassifier():
    def __init__(self, X, T, debug):
        self._X = X
        self._T = T
        self._debug = debug

    #
    # Build Classifier
    #
    def build(self):
        print ("HistogramClassifier: building classifier")

        x_size  = np.shape(self._X); # dimensions of input array
        self._labels = get_labels(self._T); # get labels
        num_features = x_size[1] # get number of features

        if self._debug == "yes":
            print ("num_features:", num_features)

        self._B = 9#bin_size(num_features) # get bin number
        self._H = hist_preprocess(self._B, num_features, self._labels) # histogram dictionary
        self._feature_min, self._feature_max = min_max(self._X, num_features) # min and max for each feature

        ### Train histogram
        histogram_train(self._X, self._T, self._B, self._H, self._feature_min, self._feature_max)

        if self._debug == "yes":
            print ("shape(B):", np.shape(self._B))
            print ("shape(H):", np.shape(self._H))
            print ("len(feature_min):", len(self._feature_min))
            print ("len(feature_max):", len(self._feature_max))

    #
    # Apply Classifier
    #
    def classify(self, Xtest):
        print ("HistogramClassifier: applying classifier")
        ### Test histogram

        Tresults = []
        for i in range(len(Xtest)):
            queries = Xtest[i] # feature vector

            hist_dic, hist_label = histogram_class(queries, self._labels, self._B, self._H, self._feature_min, self._feature_max)
            if self._debug == "yes":
                print ("Histogram:", hist_label)
            Tresults.append(hist_label[0])


        return Tresults







''' IMPLEMENTATION EXAMPLE  

### Read Data

X,T = read_letter_recognition_dataset('./datasets/letter-recognition/letter-recognition.data')

### Set parameters

x_size  = np.shape(X); # dimensions of input array
labels = get_labels(T); # get labels
no_features = x_size[1] # get number of features
B = bin_size(no_features) # get bin number
H = hist_preprocess(B,no_features,labels) # histogram dictionary
feature_min, feature_max = min_max(X,no_features) # min and min for each feature

### Train histogram
histogram_train(X,T,H,B,feature_min,feature_max)

### Test histogram
idx = 11
queries = X[idx] # feature vector
label_q = T[idx] # label of query
hist_dic, hist_label = histogram_class(queries,labels,B,feature_min,feature_max)
print ("Histogram:",hist_label,"True label:",label_q)

'''
