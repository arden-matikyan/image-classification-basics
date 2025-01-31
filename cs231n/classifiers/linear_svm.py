from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    # initialize the gradient matrix as zero, same shape as W
    dW = np.zeros(W.shape)  

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    # for every sample in X (batch of data samples)
    for i in range(num_train):
           
        # Calculate scores for training sample, will be vector of size # of classes 
        scores = X[i].dot(W)
        # X[i]: (3073,) x W: (3073, 10)
        
        # score of the true sample 
        correct_class_score = scores[y[i]]
        # print(i, ": ", correct_class_score)

        # now we iterate through all classes and ignore the true class for the current sample i 
        for j in range(num_classes):
            
            # print(j, ": ", y[i])
            if j == y[i]:
                continue

            # calculate margin to add to loss, current sample score must be greater than true class score by more than one 
            # (hinge loss function)
            margin = scores[j] - correct_class_score + 1  # note delta = 1

            # Li = sum{j!=yi}(max(0,sj-syi+1))
            if margin > 0:
                loss += margin
                
                # lim h-> 0 {   
                dW[:,j] += X[i] # + f(x+h)
                dW[:,y[i]] -= X[i] # - f(x) } 

            # (else)
            # if the margin is less than or equal to 0, it is not included in the loss sum 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # gradient dW formula /h 
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    # compute score matrix sa
    scores = np.matmul(X,W)

    # calculate margin with delta = 1 and compute hinge loss function 
    maxmargin = scores - scores[range(num_train), y].reshape(-1, 1) + 1
    margins = np.maximum(0, maxmargin)

    # set correct class margins to 0 
    margins[range(num_train), y] = 0
    # sum total loss and regularize 
    loss = np.sum(margins) / num_train + reg * np.sum(W ** 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # create a mask of zeros the same size of margins, fill in all entries to one 
    # where the margins value is > 0. Points where the margin is <= 0 will not influence the gradient 
    mask = [0]*margins
    for i in range(margins.shape[0]):
      for j in range(margins.shape[1]):
        if margins[i][j] > 0:
          mask[i][j] = 1

    # for each training example (across num_train) subtract the sum of the mask row for the correct class 
    mask[range(num_train), y] = -np.sum(mask, axis=1)
    
    # calculate gradient and add regularization 
    dW = np.matmul(X.T,mask) / num_train + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
