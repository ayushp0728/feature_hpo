
import sys
import pickle

# Parse args
n_samples_to_label = 10
for i, arg in enumerate(sys.argv):
    if arg == "--n_samples" and i + 1 < len(sys.argv):
        n_samples_to_label = int(sys.argv[i + 1])

# Load current data state
try:
    with open('al_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"[Simulation] Data loaded: {len(data['labeled_indices'])} labeled samples")
except:
    print("[Simulation] No data found, run prepare_data.py first")
    sys.exit(1)
