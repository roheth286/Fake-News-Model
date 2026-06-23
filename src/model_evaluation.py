from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Prevent linter errors in IDE when analyzed in isolation
if 'log_reg_model' not in globals():
    from sklearn.linear_model import LogisticRegression
    log_reg_model = LogisticRegression()
if 'X_val_tfidf' not in globals():
    import scipy.sparse
    X_val_tfidf = scipy.sparse.csr_matrix((0, 0))
if 'y_val' not in globals():
    import numpy as np
    y_val = np.array([])

val_predictions = log_reg_model.predict(X_val_tfidf)
val_probabilities = log_reg_model.predict_proba(X_val_tfidf)[:, 1]

accuracy = accuracy_score(y_val, val_predictions)
report = classification_report(y_val, val_predictions)

fpr, tpr, _ = roc_curve(y_val, val_probabilities)
roc_auc = auc(fpr, tpr)

print("Model Performance on Validation Data:")
print("=====================================")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# Save the plot to the output folder or workspace root if needed, but display it here
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Fake News Detection')
plt.legend(loc='lower right')
plt.show()
