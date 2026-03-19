
import sys
import pickle
import json
import numpy as np

# Parse args
n_select = 10
for i, arg in enumerate(sys.argv):
    if arg == "--n_select" and i + 1 < len(sys.argv):
        n_select = int(sys.argv[i + 1])

print(f"[ActiveLearn] Selecting {n_select} most uncertain samples...")

# Load data and model
with open('al_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

X_train = data['X_train']
unlabeled_indices = data['unlabeled_indices']

if len(unlabeled_indices) == 0:
    print("No unlabeled samples remaining!")
    sys.exit(0)

# Get predictions on unlabeled pool
X_unlabeled = X_train[unlabeled_indices]
proba = clf.predict_proba(X_unlabeled)

# Uncertainty sampling: select samples with highest entropy
entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
most_uncertain_idx = np.argsort(entropy)[-n_select:]

# Get actual indices in training set
selected_indices = unlabeled_indices[most_uncertain_idx]

# Update labeled/unlabeled sets
data['labeled_indices'] = np.concatenate([data['labeled_indices'], selected_indices])
data['unlabeled_indices'] = np.delete(unlabeled_indices, most_uncertain_idx)

# Save updated data
with open('al_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Selected {len(selected_indices)} samples")
print(f"Now: {len(data['labeled_indices'])} labeled, {len(data['unlabeled_indices'])} unlabeled")

# Save result
result = {
    'selected': int(len(selected_indices)),
    'n_labeled': int(len(data['labeled_indices'])),
    'n_unlabeled': int(len(data['unlabeled_indices']))
}

with open('acl_result.json', 'w') as f:
    json.dump(result, f)
