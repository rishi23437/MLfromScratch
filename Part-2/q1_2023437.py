import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import math

# DATA PREPROCESSING
train_images = idx2numpy.convert_from_file('train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
test_images  = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
test_labels  = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

train_mask = np.isin(train_labels, [0, 1])
test_mask  = np.isin(test_labels,  [0, 1])

images_train = train_images[train_mask]
labels_train = train_labels[train_mask]
images_test  = test_images[test_mask]
labels_test  = test_labels[test_mask]

# randomly selecting 1000 samples of each class
idx0 = np.where(labels_train == 0)[0]
idx1 = np.where(labels_train == 1)[0]

selected_samples0 = np.random.choice(idx0, size=1000, replace=False)
selected_samples1 = np.random.choice(idx1, size=1000, replace=False)
sel  = np.concatenate([selected_samples0, selected_samples1])

X_train = images_train[sel]
y_train = labels_train[sel]

X_test = images_test
y_test = labels_test

X_train = X_train.reshape(len(X_train), -1)
X_test  = X_test.reshape(len(X_test),   -1)

pca = PCA(n_components=5)                                       # extracting first 5 dimensions with max eigenvalues
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# X_train, y_train currently have 2000 examples each
X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2,        # 20% → 400 samples
    random_state=42,
    stratify=y_train      # keep class‐balance in both splits
)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# use X_train, y_train for training AdaBoost model, and X_test,  y_test for testing

######################################################################################################

'''
ADA BOOST
'''

def h(X, y, W, num_cuts = 3):
    """
    Find the best weighted decision stump.

    Inputs: X, y, W: X_train, y_train, weights. num_cuts: evaluate between 3 cuts per dimension
    Returns: 
        h(): dictionary containing {dimension, cut, sign}. dimension: jis dimension pe cut maara hai

    """
    n, p = X.shape
    W = W.astype(float)
    Wsum = W.sum()

    best_loss = np.inf
    best_stump = {}
    best_preds = None

    for j in range(p):
        col = X[:, j]
        lo, hi = col.min(), col.max()

        cuts = np.linspace(lo, hi, num_cuts+2)[1:-1]              # picking 3 cuts uniformly for each dimension

        for cut_point in cuts:
            # we can predict +1 in eithr parts of the cutoff(cut_point). so, evaluating both
            preds_pos = (col >= cut_point).astype(int)
            preds_neg = (col < cut_point).astype(int)

            for sign, preds in [(+1, preds_pos), (-1, preds_neg)]:

                loss = (W * (preds != y)).sum() / Wsum

                if loss < best_loss:
                    # we have found a better weak learner. update the best stump
                    best_loss = loss
                    best_stump = {
                        'dimension': j,
                        'cut_point': cut_point,
                        'sign': sign,
                        'loss': loss
                    }
                    best_preds = preds.copy()

    return best_stump, best_preds, best_loss


def adaboost(y, curr_stump, curr_preds, curr_loss, curr_weights, boosted_classifier):
    '''
    Run this in each iteration of adaboost
    boosted_classifier: pairs of (alpha, stump)
    '''
    alpha = (1/2)*math.log((1-curr_loss)/curr_loss)
    alpha = min(alpha, 1.0)                                             # capping alpha at 1
    boosted_classifier.append((alpha, curr_stump))

    new_weights = curr_weights.copy()
    for i in range(len(y)):
        if y[i] != curr_preds[i]:
            # updating weights of misclassified samples
            new_weights[i] *= np.exp(2*alpha)

    return new_weights, boosted_classifier

def predict_ensemble(x_test, ensemble):
    total = 0.0
    for alpha, stump in ensemble:
        dimension = stump['dimension']
        theta = stump['cut_point']
        sign = stump['sign']
        pred = int((x_test[dimension] >= theta)) if sign== +1 else int((x_test[dimension] < theta))
        total += alpha * (1 if pred == 1 else -1)
    return 1 if total > 0 else 0

def eval_ensemble(X, y_pm, y_raw, ensemble):
    F = np.zeros(X.shape[0])
    for alpha, stump in ensemble:
        j, theta, sign = stump['dimension'], stump['cut_point'], stump['sign']
        h01 = (X[:, j] >= theta).astype(int)
        if sign == -1:
            h01 = 1 - h01
        h_pm = np.where(h01==1, +1, -1)                     # 0 -> -1
        F += alpha * h_pm

    loss = np.mean(np.exp(-y_pm * F))
    
    y_pred = (F > 0).astype(int)
    err    = np.mean(y_pred != y_raw)
    return loss, err


# MAIN

y_tr_pm   = np.where(y_train == 0, -1, +1)
y_val_pm  = np.where(y_val   == 0, -1, +1)
y_test_pm = np.where(y_test  == 0, -1, +1)

M = 150
train_losses = []
val_losses   = []
test_losses  = []
train_errs   = []

W = np.array([1/len(X_train) for i in range(len(X_train))])       # Weights for samples. initially, each weight = 1/2000(2000 samples)
ensemble = []

for i in range(M):                                                    # 150 boosting rounds
    stump_i, preds_i, loss_i = h(X_train, y_train, W)
    W, ensemble = adaboost(y_train, stump_i, preds_i, loss_i, W, ensemble)

    # train, val and test losses for plotting
    train_loss, train_err = eval_ensemble(X_train, y_tr_pm,   y_train, ensemble)
    val_loss, _ = eval_ensemble(X_val,   y_val_pm,  y_val,   ensemble)
    test_loss, _ = eval_ensemble(X_test,  y_test_pm, y_test,  ensemble)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    train_errs.append(train_err)

print("Final first 5 sample weights: ", W[:5], " Final first 2 stumps and their weights: ", ensemble[:5]) # testing

preds = [predict_ensemble(test_point, ensemble) for test_point in X_test]
accuracy = np.mean(np.array(preds) == y_test)

print(f"Test set accuracy: {accuracy*100:.2f}%")

# Plot 1: Loss curves
rounds = np.arange(1, M + 1)
plt.figure()
plt.plot(rounds, train_losses)
plt.plot(rounds, val_losses)
plt.plot(rounds, test_losses)
plt.xlabel('Boosting Round')
plt.ylabel('Loss')
plt.legend(['Train','Validation','Test'])
plt.title('Train/Val/Test Loss vs. Boosting Rounds')

# Plot 2: Training misclassification error
plt.figure()
plt.plot(rounds, train_errs)
plt.xlabel('Boosting Round')
plt.ylabel('Training Error Rate')
plt.title('Training Error vs. Boosting Rounds')

plt.show()