import numpy as np

# SGD Optimizer 
class SGD:
    def __init__(self, learning_rate=0.001, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        # Initial velocity
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        # Update the Gradients
        for i in range(len(params)):
            # With momentum
            if self.momentum != 0.0:
                # Integrate velocity
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grads[i]
                
                # Integrate position
                params[i] += self.velocity[i]
            
            # Vanilla SGD
            else:
                # Update parameters
                params[i] -= self.learning_rate * grads[i]

# Adam Optimizer
class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            # Initialize moment estimates
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Calculate first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Calculate first moment estimate
            m_t = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Calculate second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Calculate second moment estimate
            v_t = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_t / (np.sqrt(v_t) + self.eps)