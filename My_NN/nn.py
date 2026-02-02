import numpy as np

class Layer_Dense:
  # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
         self.output = np.dot(inputs, self.weights) + self.biases  
    
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
        
class Softmax:
    def __init__(self):
        pass
    
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probs = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs        


class Loss:
  # Calculates the data and regularization losses
  # given model output and ground truth values
  def calculate(self, output, y):
    # Calculate sample losses
    sample_losses = self.forward(output, y)
    # Calculate mean loss
    data_loss = np.mean(sample_losses)
    # Return loss
    return data_loss
        
class Cross_Entropy_Categorical:
    """
    y_pred: softmax output
            Size: (num_batches x num_labels)
            Type: Python List or Numpy Array
    y_true: true labels
            Tyep: Python List or Numpy Array
            Size: either c1 or c2
                    c1. sparse labels: (num_batches,)
                    c2. one-hot labels: (num_batches x num_labels)
        
        
    USAGE:
    out = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
    ])

    y_true_sparse = [0, 1, 1]

    y_true_oneHot = np.array([
        [1,0,0],
        [0,1,0],
        [0,1,0]
    ])

    loss = Cross_Entropy_Categorical()
    
    loss.forward(out, y_true_sparse)
    >>> array([0.35667494, 0.69314718, 0.10536052])
    
    loss.forward(out, y_true_oneHot)
    >>> array([0.35667494, 0.69314718, 0.10536052])
    
    """
    def forward(self, y_pred, y_true):
        
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape)==1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape)==2:
            correct_confidence = np.sum(y_pred_clipped*y_true, axis=1)
        else:
            return np.nan
        
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.grad_inputs = -y_true/dvalues 
        self.grad_inputs = self.grad_inputs/samples #NORMALIZATION
        
        
class Softmax_Plus_CrossEntropyCategorical:
    def __init__(self):
        self.loss = Cross_Entropy_Categorical()
    
    def forward(self, inputs, y_true):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probs = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs
        
        return np.mean(self.loss.forward(self.output, y_true))
        
    def backward(self, dvalues, ytrue):
        samples = len(dvalues)
        
        if len(ytrue.shape) == 2:
            ytrue = np.argmax(ytrue, axis=1)
        
        self.grad_inputs = dvalues.copy()
        self.grad_inputs[range(samples), ytrue] -= 1
        
        #NORMALIZE
        self.grad_inputs = self.grad_inputs/samples        