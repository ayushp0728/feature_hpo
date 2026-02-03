"""HPO with Neural Network - Optimizing deep learning hyperparameters.

This example uses:
- Real dataset (MNIST handwritten digits via sklearn)
- Real ML model (Multi-layer Perceptron / Neural Network)
- Real active learning (uncertainty sampling with neural networks)
- Real hyperparameters to optimize (learning_rate, hidden_layers, dropout, activation)

Run from the rose/ root directory:
    python examples/hpo_neural_network.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

from rose.hpo import GridSearch, HPOConfig, HPOOrchestrator, RandomSearch
from rose.al.active_learner import ParallelActiveLearner


# ============================================================================
# Create Neural Network Active Learning Scripts
# ============================================================================


def create_nn_scripts():
    """Create Python scripts for neural network active learning workflow."""
    script_dir = Path("hpo_nn_scripts")
    script_dir.mkdir(exist_ok=True)

    # Data preparation script
    data_prep = script_dir / "prepare_data.py"
    data_prep.write_text("""
import numpy as np
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Normalize features (important for neural networks!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Start with small labeled set (5% - smaller than RF because NNs are data hungry)
n_initial = int(len(X_train) * 0.05)
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
    'unlabeled_indices': unlabeled_indices,
    'scaler': scaler
}

with open('nn_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Data prepared: {len(labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled")
print(f"Features normalized with StandardScaler")
""")

    # Simulation script
    sim_script = script_dir / "simulation.py"
    sim_script.write_text("""
import sys
import pickle

# Parse args
n_samples_to_label = 10
for i, arg in enumerate(sys.argv):
    if arg == "--n_samples" and i + 1 < len(sys.argv):
        n_samples_to_label = int(sys.argv[i + 1])

# Load current data state
try:
    with open('nn_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"[Simulation] Data loaded: {len(data['labeled_indices'])} labeled samples")
except:
    print("[Simulation] No data found, run prepare_data.py first")
    sys.exit(1)
""")

    # Training script with neural network hyperparameters
    train_script = script_dir / "training.py"
    train_script.write_text("""
import sys
import pickle
import json
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# Parse hyperparameters from command line
learning_rate = 0.001
hidden_layer_sizes = (100,)
dropout = 0.0
activation = 'relu'
max_iter = 200

for i, arg in enumerate(sys.argv):
    if arg == "--learning_rate" and i + 1 < len(sys.argv):
        learning_rate = float(sys.argv[i + 1])
    if arg == "--hidden_layers" and i + 1 < len(sys.argv):
        # Parse hidden layers: "128,64" -> (128, 64)
        layers_str = sys.argv[i + 1]
        hidden_layer_sizes = tuple(map(int, layers_str.split(',')))
    if arg == "--dropout" and i + 1 < len(sys.argv):
        dropout = float(sys.argv[i + 1])
    if arg == "--activation" and i + 1 < len(sys.argv):
        activation = sys.argv[i + 1]
    if arg == "--max_iter" and i + 1 < len(sys.argv):
        max_iter = int(sys.argv[i + 1])

print(f"[Training] lr={learning_rate}, hidden_layers={hidden_layer_sizes}, "
      f"dropout={dropout}, activation={activation}")

# Load data
with open('nn_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
labeled_indices = data['labeled_indices']

# Get labeled data only
X_labeled = X_train[labeled_indices]
y_labeled = y_train[labeled_indices]

print(f"Training neural network on {len(X_labeled)} labeled samples...")

# Train neural network with hyperparameters
# Note: sklearn's MLPClassifier doesn't have dropout built-in,
# but we simulate the effect by using early_stopping and validation_fraction
clf = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    learning_rate_init=learning_rate,
    max_iter=max_iter,
    early_stopping=True,
    validation_fraction=dropout if dropout > 0 else 0.1,
    random_state=42,
    verbose=False
)

clf.fit(X_labeled, y_labeled)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Training iterations: {clf.n_iter_}")

# Save model
with open('nn_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Save metrics
result = {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'n_labeled': len(X_labeled),
    'learning_rate': learning_rate,
    'hidden_layers': list(hidden_layer_sizes),
    'dropout': dropout,
    'activation': activation,
    'n_iter': int(clf.n_iter_)
}

with open('train_result.json', 'w') as f:
    json.dump(result, f)
""")

    # Active learning script
    acl_script = script_dir / "active_learn.py"
    acl_script.write_text("""
import sys
import pickle
import json
import numpy as np

# Parse args
n_select = 15
for i, arg in enumerate(sys.argv):
    if arg == "--n_select" and i + 1 < len(sys.argv):
        n_select = int(sys.argv[i + 1])

print(f"[ActiveLearn] Selecting {n_select} most uncertain samples...")

# Load data and model
with open('nn_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('nn_model.pkl', 'rb') as f:
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
with open('nn_data.pkl', 'wb') as f:
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
""")

    return script_dir


# ============================================================================
# Task Functions
# ============================================================================


async def simulation_task(*args, **kwargs):
    """Simulation task."""
    n_samples = kwargs.get("--n_samples", 15)
    work_dir = kwargs.get("--work_dir", ".")
    script_dir = Path("hpo_nn_scripts").absolute()
    
    return f"cd {work_dir} && {sys.executable} {script_dir}/simulation.py --n_samples {n_samples}"


async def training_task(*args, **kwargs):
    """Training task with neural network hyperparameters."""
    learning_rate = kwargs.get("--learning_rate", 0.001)
    hidden_layers = kwargs.get("--hidden_layers", "100")
    dropout = kwargs.get("--dropout", 0.0)
    activation = kwargs.get("--activation", "relu")
    work_dir = kwargs.get("--work_dir", ".")

    script_dir = Path("hpo_nn_scripts").absolute()
    cmd = (
        f"cd {work_dir} && {sys.executable} {script_dir}/training.py "
        f"--learning_rate {learning_rate} "
        f"--hidden_layers {hidden_layers} "
        f"--dropout {dropout} "
        f"--activation {activation} "
        f"--max_iter 300"
    )
    return cmd


async def active_learn_task(*args, **kwargs):
    """Active learning task."""
    n_select = kwargs.get("--n_select", 15)
    work_dir = kwargs.get("--work_dir", ".")
    script_dir = Path("hpo_nn_scripts").absolute()
    return f"cd {work_dir} && {sys.executable} {script_dir}/active_learn.py --n_select {n_select}"


# ============================================================================
# Main HPO
# ============================================================================


async def main():
    print("=" * 70)
    print("HPO WITH NEURAL NETWORKS")
    print("=" * 70)
    print("Dataset: sklearn digits (1797 images, 64 features, 10 classes)")
    print("Model: Multi-Layer Perceptron (Neural Network)")
    print("AL Strategy: Uncertainty Sampling (entropy)")
    print("Optimizing: learning_rate, hidden_layers, dropout, activation")
    print("=" * 70)

    # Create scripts
    print("\nCreating NN scripts...")
    script_dir = create_nn_scripts()
    print(f"✓ Scripts created in {script_dir}/")

    # Prepare initial data
    print("\nPreparing dataset...")
    import subprocess

    subprocess.run([sys.executable, script_dir / "prepare_data.py"], check=True)
    print("✓ Data prepared")

    # Create workflow engine
    print("\nInitializing execution backend...")
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    print("✓ Engine ready")

    # Create parallel active learner
    al_learner = ParallelActiveLearner(asyncflow)

    # Register tasks
    @al_learner.simulation_task
    async def simulation(*args, **kwargs):
        return await simulation_task(*args, **kwargs)

    @al_learner.training_task
    async def training(*args, **kwargs):
        return await training_task(*args, **kwargs)

    @al_learner.active_learn_task
    async def active_learn(*args, **kwargs):
        return await active_learn_task(*args, **kwargs)

    print("✓ Tasks registered")

    # Define hyperparameter search space for neural networks
    hyperparameter_space = {
        "--learning_rate": [0.001, 0.01, 0.1],  # Learning rate
        "--hidden_layers": ["64", "128", "64,32", "128,64"],  # Network architecture
        "--dropout": [0.0, 0.1, 0.2],  # Dropout rate (simulated)
        "--activation": ["relu", "tanh"],  # Activation function
    }

    # Total combinations: 3 * 4 * 3 * 2 = 72 configurations

    print(f"\nHyperparameter search space:")
    for param, values in hyperparameter_space.items():
        print(f"  {param}: {values}")
    print(f"Total combinations: 72")

    # Create HPO configuration
    config = HPOConfig(
        hyperparameter_space=hyperparameter_space,
        trials_per_round=4,  # 4 parallel trials per round
        max_rounds=4,  # 4 rounds = 16 total trials
        max_iter_per_trial=3,  # Each trial: 3 AL iterations
        metric_name="accuracy",  # Optimize test accuracy
        metric_mode="max",  # Maximize
    )

    # Custom metric extractor
    trial_counter = [0]
    
    def extract_accuracy(result):
        """Extract accuracy from training result."""
        import json
        
        work_dir = f"trial_{trial_counter[0]}"
        result_file = Path(work_dir) / "train_result.json"
        trial_counter[0] += 1
        
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                return data.get("accuracy", 0.0)
        except Exception as e:
            print(f"Warning: Could not read {result_file}: {e}")
            return 0.0

    # Create orchestrator with RandomSearch
    print(f"\nUsing RandomSearch strategy (16 out of 72 configs)")
    orchestrator = HPOOrchestrator(
        parallel_learner=al_learner,
        search_strategy=RandomSearch(seed=42),
        config=config,
        metric_extractor=extract_accuracy,
    )

    # Run optimization
    print("\nStarting HPO optimization...")
    print("This will take 3-5 minutes...\n")

    result = await orchestrator.optimize()

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(result.summary())

    print(f"\nBest hyperparameters:")
    for param, value in result.best_config.items():
        print(f"  {param}: {value}")

    print(f"\nTop 5 configurations:")
    top_configs = orchestrator.get_top_k_configs(k=5)
    for i, cfg in enumerate(top_configs, 1):
        trials = [t for t in result.all_trials if t.hyperparameters == cfg]
        if trials:
            metric = trials[0].metric_value
            print(f"  {i}. Accuracy={metric:.4f}: {cfg}")

    # Statistics
    stats = orchestrator.get_statistics()
    print(f"\nStatistics across all trials:")
    print(f"  Mean accuracy: {stats['mean']:.4f}")
    print(f"  Std accuracy:  {stats['std']:.4f}")
    print(f"  Min accuracy:  {stats['min']:.4f}")
    print(f"  Max accuracy:  {stats['max']:.4f}")

    print("=" * 70)

    # Cleanup
    await engine.shutdown()

    # Optional: keep scripts and data for inspection
    print(f"\n✓ Complete! Scripts and data saved in {script_dir}/")
    print(f"  - Try manually: python {script_dir}/training.py --learning_rate 0.01 --hidden_layers 128,64")


if __name__ == "__main__":
    asyncio.run(main())