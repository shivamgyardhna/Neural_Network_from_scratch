import numpy as np
# Stochastic Gradient Descent
class SGD:
  def __init__(self, learning_rate=0.5):
    self.learning_rate = learning_rate

  def update_params(self, layer):
    layer.weights += -self.learning_rate * layer.dweights
    layer.biaseses += -self.learning_rate * layer.dbiases
    


class Optimizer_variable_decay:
    def __init__(self, lr=1, decay=0, momentum=0):
        self.learning_rate = lr
        self.current_learning_rate = lr
        self.decay = decay
        self.iterations = 0
        
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate/(1+(self.decay*self.iterations))      
            
      
    def update_params(self, layer):      
        layer.weights += -self.current_learning_rate*layer.dweights
        layer.biases += -self.current_learning_rate*layer.dbiases   
        
        
    def post_update_params(self):
        self.iterations += 1



# Gradient Descent with  momentum
class Optimizer_Momentum:
    def __init__(self, lr=1, decay=0, momentum=0):
        self.learning_rate = lr
        self.current_learning_rate = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate/(1+(self.decay*self.iterations))      
            
      
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.biases_momentum = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum*layer.weight_momentum - self.current_learning_rate*layer.dweights
            layer.weight_momentum =  weight_updates
            
            biases_updates = self.momentum*layer.biases_momentum - self.current_learning_rate*layer.dbiases
            layer.biases_momentum =  biases_updates
        
        #simple Gradient Descent    
        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            biases_updates = -self.current_learning_rate*layer.dbiases
            
        
        layer.weights += weight_updates
        layer.biases += biases_updates   
        
        
    def post_update_params(self):
        self.iterations += 1

class Optimizer_AdaGrad:
    def __init__(self, lr=1, decay=0, epsilon=1e-7):
        self.learning_rate = lr
        self.current_learning_rate = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate/(1+(self.decay*self.iterations))      
            
      
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)
            
        layer.weight_cache += layer.dweights**2
        layer.biases_cache += layer.dbiases**2
        
        layer.weights += -(self.current_learning_rate*layer.dweights)/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -(self.current_learning_rate*layer.dbiases)/(np.sqrt(layer.biases_cache)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMS_PROP:
    def __init__(self, lr=1, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = lr
        self.current_learning_rate = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate/(1+(self.decay*self.iterations))      
            
      
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)
            
        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho)*(layer.dweights**2)
        layer.biases_cache = self.rho*layer.biases_cache + (1-self.rho)*(layer.dbiases**2)
        
        layer.weights += -(self.current_learning_rate*layer.dweights)/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -(self.current_learning_rate*layer.dbiases)/(np.sqrt(layer.biases_cache)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Adam_Optimizer:
    def __init__(self, lr=0.001, decay=0, epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = lr
        self.current_learning_rate = lr
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, "weight_momentums"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.beta1 * layer.weight_momentums
            + (1 - self.beta1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta1 * layer.bias_momentums
            + (1 - self.beta1) * layer.dbiases
        )

        weight_momentums_corrected = (
            layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        )
        bias_momentums_corrected = (
            layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))
        )

        layer.weight_cache = (
            self.beta2 * layer.weight_cache
            + (1 - self.beta2) * (layer.dweights ** 2)
        )
        layer.biases_cache = (
            self.beta2 * layer.biases_cache
            + (1 - self.beta2) * (layer.dbiases ** 2)
        )

        weight_cache_corrected = (
            layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        )
        biases_cache_corrected = (
            layer.biases_cache / (1 - self.beta2 ** (self.iterations + 1))
        )

        layer.weights -= (
            self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases -= (
            self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(biases_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1
    