import os
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from utils import *
from Optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

# Datasets
datasets = [
    "./datasets/credit_approval.csv",
    "./datasets/parkinsons.csv",
    "./datasets/rice.csv",
    "digits",
]

# Best settings (hidden_layers, reg, lr, batch)
best_settings = {
    "./datasets/credit_approval.csv":   ([64, 64, 64], 8e-4, 1e-3, 16),
    "./datasets/parkinsons.csv":        ([64, 64, 64], 7e-4, 1e-3, 8),
    "./datasets/rice.csv":              ([64, 64], 0.0, 3e-3, 100),
    "digits":                           ([128, 64, 32], 1e-3, 3e-3, 100),
}

# Training hyperparameters
num_epochs = 50

# Output directory
out_dir = "learning_curves"
os.makedirs(out_dir, exist_ok=True)

# Create learning curve graphs
for dataset in datasets:
    # Load settings
    hidden_layers, reg, lr, bs = best_settings[dataset]

    # Load and preprocess the data
    if dataset == "digits":
        name = "digits"
        data = load_digits()
        X = data.data.astype(float) / 16.0
        Y_unprocessed = data.target
        encoder = OneHotEncoder(sparse_output=False)
        Y = encoder.fit_transform(Y_unprocessed.reshape(-1,1))
    else:
        name, _ = os.path.splitext(os.path.basename(dataset))
        X_unprocessed, Y_unprocessed, num_attr, cat_attr = load_dataset(dataset)
        X, Y = preprocess_dataset(X_unprocessed, Y_unprocessed, num_attr, cat_attr)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=(Y_unprocessed))
    N = X_train.shape[0]

    # Build the network
    dims = [X.shape[1]] + hidden_layers + [Y.shape[1]]
    model = NeuralNetwork(dims, reg=reg)
    optimizer = Adam(learning_rate=lr)

    # Train
    epoch_losses, test_losses, _, _ = model.train(
        X_train, X_val,
        y_train, y_val,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=bs,
        verbose=False
    )

    # Plot J-curve
    num_train = np.arange(1, num_epochs+1) * N
    plt.figure()
    plt.plot(num_train, test_losses)
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Performance $J$")
    plt.title(f"Learning Curve for {name} (lr={lr})")
    plt.grid(True)

    # Save figure
    out_path = os.path.join(out_dir, f"learning_curve_{name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"File location at: {out_path}")
