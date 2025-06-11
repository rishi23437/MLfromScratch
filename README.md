# MLfromScratch
This repo is a collection of “from-scratch” implementations (ie., without using any ML modules) for my course on Statistical Machine Learning.

---

## Part 1

1. **Q1 & Q2: Decision Tree, Bagging & Random Forest**  
   - Implemented a binary-splitting Decision Tree Classifier using the Gini index.  
   - Extended to Bagging and a Random Forest ensemble.  
   - Dataset snapshot:  
     ![Dataset](https://github.com/user-attachments/assets/6b1d5a7e-0b1a-408c-b06e-8d08b3a1114e)

2. **Q3: Cross-Validation**  
   - Performed 5-fold CV on a regression model approximating \(y = \sin x\).  
   - Evaluated fold-wise MSE.

---

## Part 2

1. **Q1: AdaBoost**  
   - Binary classification on MNIST (digits 0 vs 1).  
   - Dimensionality reduction via PCA to 5 principal components.  
   - Decision stumps as weak learners.

2. **Q2: Gradient Boosting**  
   - Implemented gradient boosting with decision stumps.  
   - Tested both squared loss and absolute loss on a synthetic dataset.

3. **Q3: Feedforward Neural Network**  
   - NN Structure: 2 inputs → 1 hidden neuron (sigmoid) → 1 output neuron.  
   - Trained using squared loss on a synthetic dataset.

---

## Modules
- Numpy
- Pandas
- Matplotlib

## References
- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- The lecture slides of my Statistical Machine Learning Course.
- Bishop - PRML.
