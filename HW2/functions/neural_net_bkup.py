import numpy as np
import matplotlib.pyplot as plt
from .layers import *


class ThreeLayerNet(object):
  """
  A three-layer fully-connected neural network. This network has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  In our case, we use the same hidden dimension across all hidden layers.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first two fully
  connected layers. In other words, the network has the following architecture:

  input - fc layer - ReLU - fc layer - ReLu - fc layer - softmax

  The outputs of the third fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-2):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, H)
    b2: Second layer biases; has shape (H,)
    W3: Second layer weights; has shape (H, C)
    b3: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in each of the hidden layers.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = std * np.random.randn(hidden_size, output_size)
    self.params['b3'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a three layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization coefficient.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    """
    print (W1.shape)
    print (b1.shape)
    print (W2.shape)
    print (b2.shape)
    print (W3.shape)
    print (b3.shape)
    """
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    N, D = X.shape

    # first FC layer
    hidden_1, _ = fc_forward(X, W1, b1)
    y_1, y1_cache = relu_forward(hidden_1)
    #hidden_1 = X.dot(W1) + b1
    #y_1 = np.maximum(0, hidden_1)

    # second FC layer
    hidden_2, _ = fc_forward(y_1, W2, b2)
    y_2, y2_cache =  relu_forward(hidden_2)
    #hidden_2 = y_1.dot(W2) + b2
    #y_2 = np.maximum(0, hidden_2)

    
    # third 
    scores, scores_cache = fc_forward(y_2, W3, b3)
    #scores = y_2.dot(W3) + b3
    #print (scores.shape)
        
    # softmax on the scores
    sum_e_scores = np.sum(np.exp(scores), axis=0)
    softmax = np.exp(scores) / sum_e_scores

    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1, W2, W3. Store the result #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5. We provided a version of softmax_loss at the  #
    # end of the file. it takes the scores and labels and computes the loss and #
    # derivatives for you.                                                      #         
    #############################################################################
    
    # softmax loss function
    sum_e_scores = np.sum(np.exp(scores), axis=0)
    softmax = np.exp(scores) / sum_e_scores
    
    """
    # Compute the loss
    Loss = -np.log(softmax.T[y,np.arange(N)])
    loss = np.sum(Loss) / N
    """
    
    # loss: Scalar giving the loss
    # dx: Gradient of the loss with respect to x
    loss, d_loss = softmax_loss(scores, y)

    # L2 regularization
    loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    C = b3.shape[0] #should be 10
    #print (C)

    one_hot_labels = np.zeros((N, C))

    for i in range(N):  
        one_hot_labels[i, y[i]] = 1

    #print (softmax.shape)
    #print (one_hot_labels.shape)
    
    #y2_cache = {}
    #y1_cache = {}
    
    # first
    dcost_dscores = softmax - one_hot_labels
    dscores_dW3 = y_2
    
    dcost_W3 = np.dot(dscores_dW3.T, dcost_dscores)
    dcost_b3 = dcost_dscores
    
    # second
    dscores_dy2 = W3
    dcost_dy2 = np.dot(dcost_dscores, dscores_dy2.T) 
    dy2_dhidden2 = relu_backward(hidden_2, y2_cache)
    dhidden2_dW2 = y_1
    
    dcost_W2 = np.dot(dhidden2_dW2.T, dy2_dhidden2 * dcost_dy2)
    dcost_b2 = dcost_dy2 * dy2_dhidden2
        
    # third
    dhidden2_dy1 = W2
    dcost_dhidden2 = dcost_dy2 * dy2_dhidden2
    dcost_dy1 = np.dot(dcost_dhidden2, dhidden2_dy1.T)
    dy1_dhidden1 = relu_backward(hidden_1, y1_cache)
    dhidden1_dW1 = X
    
    dcost_W1 = np.dot(dhidden1_dW1.T, dy1_dhidden1 * dcost_dy1)
    dcost_b1 = dcost_dy1 * dy1_dhidden1
    
    """
    print(dcost_W1.shape)
    print(dcost_b1.shape)
    print(dcost_W2.shape)
    print(dcost_b2.shape)
    print(dcost_W3.shape)
    print(dcost_b3.shape)
    """

    # update grads
    grads['W1'] = dcost_W1 + reg * W1
    grads['b1'] = dcost_b1
    
    grads['W2'] = dcost_W2 + reg * W2
    grads['b2'] = dcost_b2
    
    grads['W3'] = dcost_W3 + reg * W3
    grads['b3'] = dcost_b3
    
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training label.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch = None
        y_batch = None
        #########################################################################
        # TODO: Create a random minibatch of training data and labels, storing  #
        # them in X_batch and y_batch respectively.                             #
        #########################################################################
        batch_indices = np.random.choice(num_train, batch_size, replace=True)
        X_batch = X[batch_indices, :]
        y_batch = y[batch_indices]
        pass
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        # Compute loss and gradients using the current minibatch
        loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
        loss_history.append(loss)
        #########################################################################
        # TODO: Use the gradients in the grads dictionary to update the         #
        # parameters of the network (stored in the dictionary self.params)      #
        # using stochastic gradient descent. You'll need to use the gradients   #
        # stored in the grads dictionary defined above.                         #
        #########################################################################

        self.params['W1'] = self.params['W1'] - learning_rate*grads['W1']
        self.params['b1'] = self.params['b1'] - learning_rate*grads['b1']
        self.params['W2'] = self.params['W2'] - learning_rate*grads['W2']
        self.params['b2'] = self.params['b2'] - learning_rate*grads['b2']
        self.params['W3'] = self.params['W3'] - learning_rate*grads['W3']
        self.params['b3'] = self.params['b3'] - learning_rate*grads['b3']

        pass
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        if verbose and it % 100 == 0:
            print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

        # Every epoch, check train and val accuracy.
        if it % iterations_per_epoch == 0:
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this three-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: data points to classify.

    Returns:
    - y_pred: predicted labels
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    fc_1 = X.dot(self.params['W1']) + self.params['b1']
    relu_1 = np.maximum(0, fc_1)
    fc_2 = np.dot(relu_1, self.params['W2']) + self.params['b2']
    relu_2 = np.maximum(0, fc_2)
    scores = np.dot(relu_2, self.params['W3']) + self.params['b3']
    sum_e_scores = np.sum(np.exp(scores), axis=0)
    softmax = np.exp(scores) / sum_e_scores	
    y_pred = np.argmax(softmax, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    return y_pred

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data
    - y: Vector of labels

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx



