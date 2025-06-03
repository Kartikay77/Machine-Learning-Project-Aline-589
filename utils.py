import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Neural Network Layer Functions
# Helper function to calculate the forward pass
def forward(X, theta):
    # Ensure that X is a 2D array for vectorized operations
    X = np.atleast_2d(X)
    
    # Add a bias column to the input matrix X: (N, D) => (N, D+1)
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    # Calculate the weighted sum Z: (N, D+1) \cdot (D+1, M) => (N, M)
    Z = X_bias.dot(theta.T)

    # Calculate the sigmoid activation A: sigmoid(Z) = 1 / (1 + exp(-Z))
    out = 1.0 / (1.0 + np.exp(-Z))
    
    # Cache the intermediate values for the backward pass
    cache = (X_bias, theta, Z)
     
    return out, cache

# Helper function to calculate the backward pass
def backward(dout, cache):
    # Unpack the cache
    X_bias, theta, _, = cache

    # Ensure that dout is a 2D array for vectorized operations
    dout = np.atleast_2d(dout)
    
    # Calculate the gradients
    # dX: (N, M) \cdot (M, D+1) => (N, D+1) => (N, D)
    dX = (dout.dot(theta) * (X_bias * (1.0 - X_bias)))[:,1:]
    
    # dtheta: (M, N) \cdot (N, D+1) => (M, D+1) 
    dtheta = dout.T.dot(X_bias) / X_bias.shape[0]
    
    return dX, dtheta

# Helper function to calculate the cross entropy loss
def cross_entropy_loss(out, y):
    # Ensure that y is a 2D array for vectorized operations   
    y = np.atleast_2d(y)
     
    # Calculate the cross entropy loss
    loss = -np.sum(y * np.log(out + 1e-8) + (1 - y) * np.log(1 - out + 1e-8))
    
    # Calculate the gradient of the loss with respect to the output
    dout = (out - y)
    
    # Return the unregularized loss and gradient
    return loss, dout

# Helper function to calculate the softmax loss and gradient
def softmax_loss(scores, y):
    N = scores.shape[0]
    
    # Convert one-hot labels to integer indices
    y = np.array(y)
    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    # Shift scores for numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # Calculate softmax probabilities
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # Compute loss
    loss = -np.sum(np.log(probabilities[np.arange(N), y] + 1e-8)) / N

    # Compute gradient
    dout = probabilities.copy()
    dout[np.arange(N), y] -= 1
    dout /= N

    # Return the loss and gradient
    return loss, dout

# Helper function to load the dataset
def load_dataset(filepath):
    # Load the dataset and shuffle it
    dataset = np.genfromtxt(filepath, delimiter=",", dtype=str)
    attributes = dataset[0, :-1]
    dataset = shuffle(dataset[1:])

    # Split the dataset into features and labels
    X = dataset[:, :-1]
    Y = dataset[:, -1]

    # Extract each type of attribute
    num_attributes = [i for i, attr in enumerate(attributes) if attr.endswith("_num")]
    cat_attributes = [i for i, attr in enumerate(attributes) if attr.endswith("_cat")]

    # Return the features, labels, and attribute type indices
    return X, Y, num_attributes, cat_attributes

# Helper function to preprocess the dataset
def preprocess_dataset(X, Y, num_attributes, cat_attributes):
    # Process numerical and categorical attributes
    X_processed = []

    # Normalize numerical attributes
    if num_attributes:
        attributes = X[:, num_attributes].astype(float)
        min = attributes.min(axis=0)
        max = attributes.max(axis=0)
        X_processed.append((attributes - min) / (max - min + 1e-8))

    # One-hot encode categorical attributes
    if cat_attributes:
        transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(sparse_output=False), cat_attributes)
            ],
            remainder='drop' 
        )
        X_processed.append(transformer.fit_transform(X))

    # Stack the processed features
    X_processed = np.hstack(X_processed).astype(float)

    # One-hot encode labels
    label_encoder = OneHotEncoder(sparse_output=False)
    Y_processed = label_encoder.fit_transform(Y.reshape(-1, 1)).astype(float)

    # Return the processed features and labels
    return X_processed, Y_processed

# Helper function to perform stratified k-fold cross-validation
def stratified_k_folds(X, Y, k):
    # Split the dataset by class
    classes = {c: np.where(Y == c)[0] for c in np.unique(Y)}
    
    # For each class
    for c in classes:
        # Shuffle the data
        np.random.shuffle(classes[c])
        
        # Split it into k folds
        classes[c] = np.array_split(classes[c], k)
    
    folds = []
    
    # For each fold, split the data into training and testing sets
    for i in range(k):
        # Training fold is folds 1, ..., i-1, i+1, ..., k
        train_idx = np.concatenate(
            [classes[c][j]
             for c in classes
             for j in range(k) if j != i]
        )
        
        # Testing fold is exactly fold i
        test_idx = np.concatenate([classes[c][i] for c in classes])
        
        # Create the fold
        X_train = X[train_idx].copy()
        X_test  = X[test_idx].copy()
        y_train = Y[train_idx].copy()
        y_test  = Y[test_idx].copy()
        
        folds.append((X_train, X_test, y_train, y_test))
    
    return folds

# Evaluation metrics
def evaluation_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    f1_scores = []
    
    # Calculate the accuracy
    accuracy = np.mean(y_true == y_pred)

    # Calculate the f1 scores for each class
    for cl in classes:
        tp = np.sum((y_pred == cl) & (y_true == cl))
        fp = np.sum((y_pred == cl) & (y_true != cl))
        fn = np.sum((y_pred != cl) & (y_true == cl))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0
        f1_scores.append(f1_score)

    # Calculate the average f1 score
    f1_score = np.mean(f1_scores)
    
    return accuracy, f1_score
