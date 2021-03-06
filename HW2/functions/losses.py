import numpy as np
from random import shuffle
from random import randrange

def softmax_loss_vectorized(W, X, y, reg=0):
    """
    Softmax loss function, vectorized version. Same as HW1.
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
    num_samples = X.shape[0]
    
    scores = W.T.dot(X.T)
    sum_e_scores = np.sum(np.exp(scores), axis=0)
    softmax = np.exp(scores) / sum_e_scores
    #print (np.sum(softmax,axis=0))
    
    """
    loss = np.log(sum_e_scores)
    loss = loss - scores[y,np.arange(num_samples)]
    loss = np.sum(loss) / num_samples
    """
    
    Loss = -np.log(softmax[y,np.arange(num_samples)])
    loss = np.sum(Loss) / num_samples
    
    # L2 regularization
    loss += 0.5 * reg * np.sum(W * W)
    
    # calc dW
    softmax[y,np.arange(num_samples)] += -1.0
    dW = softmax.dot(X).T
    dW /= num_samples
    dW += reg * W
    #print (dW.shape)

    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW

def grad_check(f, x, analytic_grad, num_checks=10, h=1e-5):
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evaluate f(x + h)
        x[ix] = oldval - h # increment by h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print ('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))


