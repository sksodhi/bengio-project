from __future__ import division

import numpy as np
import numpy.linalg as LA


"""
" This function performs PCA on given matrix X.
" It takes two inputs:
" X : Matrix X on which to perform PCA
" num_of_dim : Number of colums of principal components to return
"
" Sample invocation :
" X,T = read_letter_recognition_dataset('./datasets/letter-recognition/letter-recognition.data')
" X_afterPCA=perform_pca(X,4)
"

"""

def perform_pca(X,num_of_dim=2):

     print "************Performing PCA with number of dimensions as %d ************"%num_of_dim

     num_of_cols_in_X=len(X[0])

     if num_of_cols_in_X < num_of_dim:
          print "Number of dimension exceed maximum colums in X; Returning all data"
          num_of_dim=num_of_cols_in_X

          
     
     mu=np.mean(X,axis=0);

     """
     " Code to check mean
     """
     
     """
     print "*****************MEAN******************"
     print mu
     print type(mu)
     print mu.shape

     """
     
     """
     " Code to find Z
     """
       
     Z=X-mu

    
     """
     " Code to verify Z
     """

     """
     print "*******************Z*******************"
     print Z.shape
     meanZ=np.mean(Z,axis=0)

     #Mean should be 0
     print meanZ

     """
     
     """
     " Code to find C
     """

     C=np.cov(Z,rowvar=False);
    
 
     """
     " Code to verify C
     """

     """
     print "*******************C*******************"
     print C.shape
     tranposeC=C.transpose()

     #C Transpose should be same as C
     print C==tranposeC

     """
     
     """
     " Code to find V
     """

     [L,V]=LA.eigh(C);


     L=np.flipud(L);
     V=np.flipud(V.T);
    
     """
     " Code to verify V
     """

     """
     row=V[0,:];col=V[:,0]

     # Should be nearing 0
     print "*******************V*******************"
     print (np.allclose(np.dot(C,row),L[0]*row))

     """
     
     """
     " Code to verify P
     """

     P=np.dot(Z,V.T);

     
     print "Dimensions before PCA: ",P.shape
     

     """
     " Code to verify P
     """

     """
     print "*******************P*******************"
     print P

     print P.shape
     """
          
     return P[:,:num_of_dim]

     
    
    
if __name__ == "__main__":
     
    """
    X,T = read_letter_recognition_dataset('./datasets/letter-recognition/letter-recognition.data')
    X_afterPCA=perform_pca(X,4)
    print X_afterPCA.shape
    """

   
   
    
    

    

