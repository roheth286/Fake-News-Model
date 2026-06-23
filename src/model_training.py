from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Prevent linter errors in IDE when analyzed in isolation
if 'X_train_tfidf' not in globals():
    import scipy.sparse
    X_train_tfidf = scipy.sparse.csr_matrix((0, 0))
if 'y_train' not in globals():
    import numpy as np
    y_train = np.array([])

X_train_split, X_val_tfidf, y_train_split, y_val = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42
)

# Fix: Define the missing log_reg_model variable which was not instantiated in the original notebook cells
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train_split, y_train_split)

print("Model training complete.")
