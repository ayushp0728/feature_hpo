
import sys
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Parse hyperparameters from command line
n_estimators = 100
max_depth = None
min_samples_split = 2
min_samples_leaf = 1

for i, arg in enumerate(sys.argv):
    if arg == "--n_estimators" and i + 1 < len(sys.argv):
        n_estimators = int(sys.argv[i + 1])
    if arg == "--max_depth" and i + 1 < len(sys.argv):
        value = sys.argv[i + 1]
        max_depth = None if value == "None" else int(value)
    if arg == "--min_samples_split" and i + 1 < len(sys.argv):
        min_samples_split = int(sys.argv[i + 1])
    if arg == "--min_samples_leaf" and i + 1 < len(sys.argv):
        min_samples_leaf = int(sys.argv[i + 1])

print(f"[Training] n_estimators={n_estimators}, max_depth={max_depth}, "
      f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

# Load data
with open('al_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
labeled_indices = data['labeled_indices']

# Get labeled data only
X_labeled = X_train[labeled_indices]
y_labeled = y_train[labeled_indices]

print(f"Training on {len(X_labeled)} labeled samples...")

# Train model with hyperparameters
clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_labeled, y_labeled)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Save metrics
result = {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'n_labeled': len(X_labeled),
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

with open('train_result.json', 'w') as f:
    json.dump(result, f)
