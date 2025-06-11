import numpy as np
from sklearn.model_selection import train_test_split

# sigmoid is the activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# GENERATING DATA --------------------------------------------------------------------------------------------

N_per_class = 10
mean0 = np.array([-1, -1])
mean1 = np.array([ 1,  1])
X0 = np.random.randn(N_per_class, 2) + mean0
X1 = np.random.randn(N_per_class, 2) + mean1
y0 = np.zeros(N_per_class)
y1 = np.ones(N_per_class)
X = np.vstack([X0, X1])
y = np.concatenate([y0, y1])

# split into train and test set, 50% each
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify = y, random_state = 0)

# randomly initializing weights
rng = np.random.default_rng(1)
W1 = rng.normal(size=(1, 2))
b1 = rng.normal(size=(1,))     
W2 = rng.normal(size=(1, 1))
b2 = rng.normal(size=(1,))

eta = 0.1                                                   # learning rate
epochs = 1000

# MAIN CODE BLOCK: Gradient Descent ------------------------------------------------------------------------

N = len(y_train)
for epoch in range(epochs):
    # Forward pass
    z1 = X_train.dot(W1.T) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2.T) + b2
    y_hat = z2[:, 0]

    error = y_hat - y_train
    squared_loss = np.mean(error**2)

    # Back propagation
    # The next line is just the derivative of the loss wrt z2: (1/N)*(y_hat - y)
    dL_dz2 = (error / N).reshape(-1, 1)

    # Gradients for W2, b2
    dW2 = dL_dz2.T.dot(a1)
    db2 = dL_dz2.sum(axis=0)

    # derivative of a1, and then z1(Hidden Layer)
    dL_da1 = dL_dz2.dot(W2)
    dL_dz1 = dL_da1 * a1 * (1 - a1)

    # Gradients for W1, b1
    dW1 = dL_dz1.T.dot(X_train)
    db1 = dL_dz1.sum(axis=0)

    # UPDATING PARAMS USING GRADIENT DESCENT
    W2 -= eta * dW2
    b2 -= eta * db2
    W1 -= eta * dW1
    b1 -= eta * db1


# Evaluating on test set
z1_test = X_test.dot(W1.T) + b1
a1_test = sigmoid(z1_test)
z2_test = a1_test.dot(W2.T) + b2
y_pred_test = z2_test[:, 0]

mse_test = np.mean((y_pred_test - y_test)**2)
print(f"Test MSE: {mse_test*100:.2f}%")
