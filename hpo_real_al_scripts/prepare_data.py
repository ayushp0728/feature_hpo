
import numpy as np
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Start with small labeled set (10%)
n_initial = int(len(X_train) * 0.1)
indices = np.random.RandomState(42).permutation(len(X_train))

labeled_indices = indices[:n_initial]
unlabeled_indices = indices[n_initial:]

# Save data
data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'labeled_indices': labeled_indices,
    'unlabeled_indices': unlabeled_indices
}

with open('al_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Data prepared: {len(labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled")
