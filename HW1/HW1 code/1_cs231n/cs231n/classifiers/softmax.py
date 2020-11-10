import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
    
    """
    Softmax loss function, vectorized version.
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_sample = len(y)
    
    score = np.dot(W,X) # K x N
    score = score - np.amax(score, axis = 0)
    #score += 1
    sum_exp = np.sum(np.exp(score), axis=0) # 1 x N
    true_exp = np.exp(score[y,range(num_sample)])
    #print(sum_exp)
    
    
    aaa = np.log(true_exp/(sum_exp) + 0.0001)
    loss_s = - np.sum( aaa )/num_sample
    
    R = np.sum( np.square(W) )
    
    loss = loss_s + 0.5 * reg * R
    
    temp = np.exp(score)/sum_exp # K x N
    temp[list(y),range(num_sample)] += -1
    dW = np.dot(temp,X.T)/num_sample + reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
