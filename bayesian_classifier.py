#
# Bayesian Classifier
#

import numpy as np


# Utility function to subtract values in two separate lists
def subtract_lists(a, b):
    assert( len( np.shape(a) ) == len( (np.shape(b) ) ) )
    assert( np.shape(a)[0] == np.shape(b)[0] )
    retVal = a
    for i, val in enumerate(a):
            retVal[i] = a[i] - b[i]
    return retVal


# Calculate One-dimensional Gaussian Model 
def calcGM(x, mu, sigma, N): 
    one_over_sqrt2pi = 1 / 1.0 * np.sqrt(2 * np.pi)
    one_over_sigma   = 1 / 1.0 * sigma
    normfact         = one_over_sqrt2pi * one_over_sigma
    exponent         = (-0.5) * np.square( (x-mu) / (sigma) )
    retval = N * normfact * np.exp( exponent )
    return retval
    

# Calculate Multi-dimensional Gaussian Model 
# For example, to compute prob of being female given height/handspan {69, 17.5}, 

# cov_F         = np.cov(H_F,S_F)
# cov_M         = np.cov(H_M,S_M)
# muF           = [muH_F, muS_F]
# muM           = [muH_M, muS_M]
# tempf         = calcMDGM([69, 17.5], muF, cov_F, NF);
# tempm         = calcMDGM([69, 17.5], muM, cov_M, NM);
# prob_f_69_175 = tempf / (tempf + tempm)
# print "Probability prob_f_69_175 : hf / (hf + hm) = \n" + str(prob_f_69_175)

def calcMDGM(x, mu, cov, N):
    if len(x) == 1:
        retval = calcGM(x, mu, cov, N)
    else:
        assert( len( np.shape(cov) ) == 2 ) 
        assert( np.shape(cov)[0] == np.shape(cov)[1] )
        assert( len( x) == np.shape( x)[0] == np.shape(cov)[0] )
        assert( len(mu) == np.shape(mu)[0] == np.shape(cov)[0] )
        x   = np.array( x )
        x   = x.reshape(1, np.shape(x)[0])
        mu  = np.array(mu)
        mu  = mu.reshape(1, np.shape(mu)[0])
        d   = np.shape(cov)[0]
        one_over_2pi   = 1 / (2 * np.pi)
        one_over_det   = 1 / np.linalg.det(cov)
        normfact       = np.power(one_over_2pi,d/2) * np.sqrt(one_over_det)
        x_minus_mu     = subtract_lists(x, mu) 
        x_minus_mu_T   = np.transpose(x_minus_mu)
        cov_inv        = np.linalg.inv(cov)
        exponent       = (-0.5) * np.mat(x_minus_mu) * cov_inv * np.mat(x_minus_mu_T)
        retval         = N * normfact * np.exp( exponent )
    return retval






# Calculate Multi-dimensional Gaussian Model Parameters
# I.e., find the mu and cov parameters for each class
#       return the feature vectors for each class

# Inputs
# X     = Feature Vectors
# T     = Class lables for each Feature Vector

# Returns
# mu    = mu vectors for each class, in each dimension (d)
# cov   = covariance matrix for each class, dxd array
# P     = Feature vectors per class
#         E.g., P[0] are the feature vectors for the first class
#         E.g., P[0][0] are the first  features for the first class
#         E.g., P[0][1] are the second features for the first class
#         E.g., P[2][1] are the second features for the third class

def calcMDGMParams(X,T):

    num_T = len(T)
    set_T = sorted(set(T))
    X_dims = np.shape(X)[1]

    # Find the number classes, setup a place to put counts
    num_classes = len(set_T)
    class_counts = np.zeros(num_classes)

    # Find the counts for each class
    max_e = 0
    max_char = 'a'
    for e in set_T:
        num_e = 0
        for i in range(num_T):
            if T[i] == e:
                num_e  = num_e + 1
        #print "found " + str(num_e) + " " + str(e) + "\'s"
        if num_e > max_e:
            max_e = num_e
            max_char = e
        index = ord(e) - ord('A')
        class_counts[index] = num_e

    # Go through the data, and place data into
    # separate lists, for each class, for each dimension/feature
    # Calculate means and covariance params

    # Allocate 'master lists'
    P   = [ ]
    mu  = [ ]
    cov = [ ]
    N   = [ ]

    # For each class
    for e in set_T:

        # Allocate temp lists for this class's calculation
        P_e   = [ ]
        sum_e = np.zeros(X_dims)
        mu_e  = np.zeros(X_dims)

        # Make space for dimension/feature
        for d in range(X_dims):
            P_e.append([])

        for i in range(num_T):

            # Find matching class data in all the data

            if T[i] == e:
                for d in range(X_dims):
                    sum_e[d] = sum_e[d] + X[i][d]
                    P_e[d].append(X[i][d])

        class_index = ord(e) - ord('A')
        for d in range(X_dims):
            mu_e[d] = sum_e[d] / class_counts[class_index]

        # Add mu_e, cov_e, and P_e to the 'master lists'
        P.append(P_e)
        mu.append(mu_e)
        cov_e = np.cov(P_e)
        cov.append(cov_e)
        N.append(class_counts[class_index])

        #print np.shape(P_e)
        #print "mu  for " + str(e) + " is "
        #print str(mu_e)
        #print "cov for " + str(e) + " is "
        #print str(cov_e)
    #print np.shape(P)

    return P, mu, cov, N



#
# Bayesian Classifier
#
class BayesianClassifier():
    def __init__(self, X, T, debug):
        self._X = X
        self._T = T
        self._debug = debug

    #
    # Build Classifier
    #
    def build(self):
        print ("BayesianClassifier: building classifier")

        self._P, self._mu, self._cov, self._N = calcMDGMParams(self._X,self._T)

        if self._debug == "yes":
            print ("len(P):", len(self._P))
            print ("len(mu):", len(self._mu))
            print ("len(cov)", len(self._cov))
            print ("N:",self._N)

    #
    # Apply Classifier
    #
    def classify(self, Xtest):
        print ("BayesianClassifier: applying classifier")

        # Make a list of all possible classes
        set_T = sorted(set(self._T))

        # Select Data Set
        X = self._X
        X = Xtest

        # Allocate output list
        result_labels = [ ]

        # Go through the data
        
        N = np.shape(X)[0]
        for n in range(N):

            # Try the n'th letter 
            test_x = X[n]

            # Calculate numerators
            numers = []
            for l in range(len(set_T)):
                mu  = self._mu [l]
                cov = self._cov[l]
                N   = self._N  [l]
                numers.append(calcMDGM(test_x, mu, cov, N))
            denom = sum(numers)

            # Calculate numerator, prob, and find most likely letter
            max_prob = 0
            max_e    = 'A'
            for i,e in enumerate(set_T):
                
                prob = numers[i] / (denom + 1e-9)
                if prob > max_prob:
                    max_e    = e
                    max_prob = prob

            result_labels.append(max_e)

        return result_labels


 

