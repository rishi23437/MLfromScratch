import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# DATA GENERATION

X = np.random.rand(100)                                     # X ~ U[0,1]
y = np.sin(2*np.pi*X) + np.cos(2*np.pi*X) \
    + np.random.normal(0, np.sqrt(0.01), 100)

X = X.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Decision Stumps ----------------------------------------------------------------------------

def fit_stump_squared(X, r, num_cuts=20):
    """
    Fitting a stump using squared loss
    """
    thresholds = np.linspace(0, 1, num_cuts+2)[1:-1]
    best_sse = np.inf
    best = {}
    best_preds = None

    xcol = X[:,0]
    for theta in thresholds:
        left = xcol <= theta
        right = ~left
        
        # predictions: mean in each region
        c_L = r[left].mean()  if left.any()  else 0.0
        c_R = r[right].mean() if right.any() else 0.0
        
        preds = np.where(left, c_L, c_R)
        sse = ((r - preds)**2).sum()
        
        if sse < best_sse:
            best_sse = sse
            best = {'threshold': theta, 'left_val': c_L, 'right_val': c_R}
            best_preds = preds.copy()
    
    return best, best_preds

def update_F(F_train, F_test, eta, gradient_train, gradient_test):
    F_train += eta * gradient_train
    F_test  += eta * gradient_test
    return F_train, F_test
    

def fit_stump_absolute(X, r, num_cuts=20):
    """
    Fitting a stump based on absolute loss
    """
    thresholds = np.linspace(0, 1, num_cuts+2)[1:-1]
    best_absolute_loss = np.inf
    best = {}
    best_preds = None

    xcol = X[:,0]
    for theta in thresholds:
        left = xcol <= theta
        right = ~left
        
        # prediciton: median in each region
        c_L = np.median(r[left])  if left.any()  else 0.0
        c_R = np.median(r[right]) if right.any() else 0.0
        
        preds = np.where(left, c_L, c_R)
        absolute_loss = np.abs(r - preds).sum()
        
        if absolute_loss < best_absolute_loss:
            best_absolute_loss = absolute_loss
            best = {'threshold': theta, 'left_val': c_L, 'right_val': c_R}
            best_preds = preds.copy()
    
    return best, best_preds

# Gradient‐boosting ----------------------------------------------------------------

def gradient_boost(X_train, y_train, X_test, y_test, fit_stump, loss_type, M=200, eta=0.01):
    n_train = len(y_train)
    F_train = np.zeros(n_train)
    F_test  = np.zeros(len(y_test))
    ensemble = []
    
    train_losses = []
    test_losses  = []
    curves = {}
    
    grid = np.linspace(0,1,200).reshape(-1,1)
    
    for m in range(1, M+1):
        # -gradient = residual
        if loss_type == 'squared':
            residual = y_train - F_train
        else:                                                                  # absolute
            residual = np.sign(y_train - F_train)
        
        stump, pred_resid = fit_stump(X_train, residual)                       
        ensemble.append((stump, eta))
        
        left_t = (X_test[:,0] <= stump['threshold'])
        pred_test = np.where(left_t, stump['left_val'], stump['right_val']) 

        # Updating F
        F_train, F_test = update_F(F_train, F_test, eta, pred_resid, pred_test)  # updating F_train by pred_resid, F_test by pred_test
        
        # recording losses
        if loss_type == 'squared':
            train_losses.append(np.mean((y_train - F_train)**2))
            test_losses .append(np.mean((y_test  - F_test )**2))
        else:
            train_losses.append(np.mean(np.abs(y_train - F_train)))
            test_losses .append(np.mean(np.abs(y_test  - F_test )))
        
        # 6) if m in our checkpoints, record curve on grid
        if m in (1, 10, 50, 100, M):
            Fg = np.zeros(len(grid))
            for (st, η) in ensemble:
                left_g = (grid[:,0] <= st['threshold'])
                pg = np.where(left_g, st['left_val'], st['right_val'])
                Fg += η * pg
            curves[m] = Fg
    
    return train_losses, test_losses, grid[:,0], curves


# Driver Code: for Gradient BOost using squared and absolute losses ------------------------------------------------
train_sq, test_sq, xgrid, curves_sq = gradient_boost(X_train, y_train, X_test, y_test, fit_stump_squared, 'squared', M=200, eta=0.01)
train_abs, test_abs, _, curves_abs = gradient_boost(X_train, y_train, X_test, y_test, fit_stump_absolute, 'absolute', M=200, eta=0.01)

# PLOTTING predictions vs. ground truth(y) -----------------------------------------------------------------
for loss_name, curves in [('Squared-Loss', curves_sq), ('Absolute-Loss', curves_abs)]:
    plt.figure(figsize=(8,5))
    plt.scatter(X_train[:,0], y_train, s=20, alpha=0.6, label='Actual Train datapoint')
    plt.scatter(X_test[:,0],  y_test,  s=20, alpha=0.6, label='Actual Test datapoint')
    
    # 5 curves for predictions at our five checkpoints
    for m, Fg in curves.items():
        plt.plot(xgrid, Fg, label=f'iter={m}')
    
    plt.title(f'{loss_name}: model fit over iterations')
    plt.xlabel('X -> U[0, 1]')
    plt.ylabel('$y -> \sin(2\pi x)+\cos(2\pi x)+\epsilon$')
    plt.legend()
    plt.show()

# PLOTTING training‐loss over all 200 rounds 
plt.figure()
plt.plot(range(1, 201), train_sq, label='train MSE')
plt.title('Training MSE vs Iteration (Squared Loss)')
plt.xlabel('Iteration'); plt.ylabel('MSE')
plt.show()

plt.figure()
plt.plot(range(1,201), train_abs, label='train MAE', color='C1')
plt.title('Training MAE vs Iteration (Absolute Loss)')
plt.xlabel('Iteration'); plt.ylabel('MAE')
plt.show()
