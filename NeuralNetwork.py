import numpy as np
from utils import *
from Optimizers import *

class NeuralNetwork:
    def __init__(self, layer_dims, reg=0.0):
        self.reg = reg
        self.num_layers = len(layer_dims) - 1
        self.params = []

        # Initialize the weights and biases for each layer
        for i, j in zip(layer_dims[:-1], layer_dims[1:]):
            # Initialize weights to unit Gaussian of shape (i, j)
            W = np.random.normal(0.0, 1.0, (j, i))

            # Initialize biases to zero of shape (j, 1)
            b = np.zeros((j, 1))

            # Concatenate the parameters [bias W]
            self.params.append(np.hstack([b, W]))

    def loss(self, X, y=None):
        N = np.atleast_2d(X).shape[0]
        
        # Initialize input and cache stack
        out, cache_stack = X, []  

        # Forward pass
        for theta in self.params:
            out, cache = forward(out, theta)
            cache_stack.append(cache)

        # Predict output
        if y is None:
            return out

        # Calculate the loss and gradient
        data_loss, dout = softmax_loss(out, y)
        grads = []

        # Backward pass through layers
        for cache in reversed(cache_stack):
            # Unpack the cache
            _, theta, _ = cache
            dx, dtheta = backward(dout, cache)

            # Add L2 regularization gradient (skip bias term)
            dtheta[:, 1:] += self.reg * theta[:, 1:] / N
            grads.insert(0, dtheta.copy())

            # Propagate gradient
            dout = dx

        # Compute L2 regularization loss
        reg_loss = 0.5 * self.reg * sum(np.sum(W[:, 1:]**2) for W in self.params)
        loss = (data_loss + reg_loss) / N

        return loss, grads

    def train(self, X_train, X_val, y_train, y_val,
              optimizer=SGD(), 
              num_epochs=10, 
              batch_size=100,
              verbose=False, 
              print_every=10):

        num_train = np.atleast_2d(X_train).shape[0]

        # Metric caches
        batch_losses, epoch_losses, val_losses, train_accs, val_accs = [], [], [], [], [] 

        for epoch in range(num_epochs):
            # Shuffle training data
            perm = np.random.permutation(num_train)

            # Create mini-batches
            batches = [
                (X_train[perm][i:i+batch_size], y_train[perm][i:i+batch_size])
                for i in range(0, num_train, batch_size)
            ]

            # Train using the mini-batchs
            for X_batch, y_batch in batches:
                loss, grads = self.loss(X_batch, y_batch)
                optimizer.update(self.params, grads)
                
                # Update batch loss cache
                batch_losses.append(loss)

            # Calculate the average training loss
            train_loss = np.mean(batch_losses[-len(batches):])
            epoch_losses.append(train_loss)

            # Validation loss
            val_loss, _ = self.loss(X_val, y_val)
            val_losses.append(val_loss)

            # Training accuracy
            train_pred = self.predict(X_train)
            train_acc = (train_pred == np.argmax(y_train, axis=1)).mean()
            train_accs.append(train_acc)

            # Validation accuracy
            val_pred = self.predict(X_val)
            val_acc = (val_pred == np.argmax(y_val, axis=1)).mean()
            val_accs.append(val_acc)

            # Verbose output
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, "
                      f"train_acc: {train_acc:.5f}, val_acc: {val_acc:.5f}")

        return epoch_losses, val_losses, train_accs, val_accs


    def predict(self, X):
        # loss returns scores when y=None
        scores = self.loss(X)
        
        # Argmax to get exact predictions 
        predictions = np.argmax(scores, axis=1)
        
        # Return the predictions
        return predictions
        

