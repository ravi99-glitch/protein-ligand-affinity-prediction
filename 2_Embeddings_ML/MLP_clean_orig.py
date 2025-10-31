import json
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------------------------------------------------------------------
# Create output directory and setup device
# ------------------------------------------------------------------------------------------------------------------------------
output_dir = "MLP_Results"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ------------------------------------------------------------------------------------------------------------------------------
# Load Embeddings
# ------------------------------------------------------------------------------------------------------------------------------
df = pd.read_pickle("PDBbind_protein_ligands_embeddings_min_MoLFormer.pkl") 

# ------------------------------------------------------------------------------------------------------------------------------
# Train - Test Split and Validation
# ------------------------------------------------------------------------------------------------------------------------------

# Load PDBbind original json and split into train and test sets
with open("PDBbind_original_data_split.json", "r") as file:
    original_data_split = json.load(file)

train_ids = original_data_split["train"]

# Extract test IDs for different CASF sets
test_ids_dict = {
    "casf2013_indep": set(original_data_split.get("casf2013_indep", [])),
    "casf2016_indep": set(original_data_split.get("casf2016_indep", [])),
    "casf2013": set(original_data_split.get("casf2013", [])),
    "casf2016": set(original_data_split.get("casf2016", [])),
}

# Load PDBbind cleansplit json and split into train and test sets
with open("PDBbind_cleansplit_data_split.json", "r") as file:
    cleansplit_data_split = json.load(file)

train_ids_clean = cleansplit_data_split["train"]
test_ids_clean = set()
for key, value in cleansplit_data_split.items():
    if key != "train":
        test_ids_clean.update(value)

# Trainingsdata original
train_df_orig = df[df["pdb_id"].isin(train_ids)]
y_train_orig = train_df_orig["pK"]

# Feature-Matrix for Original-Trainingsset => concatenate protein and ligand embeddings
X_train_orig = np.vstack(
    train_df_orig.apply(lambda row: np.concatenate([row["protein_embedding"], row["molformer_embedding"]]), axis=1)
)

# Testdata original 
test_datasets = {}
for key, ids in original_data_split.items():
    if key != 'train':  
        test_df = df[df["pdb_id"].isin(ids)]
        y_test = test_df["pK"]
        X_test = np.vstack(
            test_df.apply(lambda row: np.concatenate([row["protein_embedding"], row["molformer_embedding"]]), axis=1)
        )
        test_datasets[key] = (X_test, y_test)

# ------------------------------------------------------------------------------------------------------------------------------

# Trainingsdata CleanSplit
train_df_clean = df[df["pdb_id"].isin(train_ids_clean)]
y_train_clean = train_df_clean["pK"]

# Feature-Matrix for CleanSplit-Trainingsset => concatenate protein and ligand embeddings 
X_train_clean = np.vstack(
    train_df_clean.apply(lambda row: np.concatenate([row["protein_embedding"], row["molformer_embedding"]]), axis=1)
)

# Testdata CleanSplit
test_datasets_clean = {}
for key, ids in cleansplit_data_split.items():
    if key != 'train': 
        test_df = df[df["pdb_id"].isin(ids)]
        y_test = test_df["pK"]
        X_test = np.vstack(
            test_df.apply(lambda row: np.concatenate([row["protein_embedding"], row["molformer_embedding"]]), axis=1)
        )
        test_datasets_clean[key] = (X_test, y_test)

# Load validation splits from JSON files
val_splits_path = "val_splits"
val_split_files = [f for f in os.listdir(val_splits_path) if f.endswith(".json")]

original_files = sorted([f for f in val_split_files if f.startswith("original")])
cleansplit_files = sorted([f for f in val_split_files if f.startswith("cleansplit")])

# Load IDs from JSONs
val_splits_original = []
for file in original_files:
    with open(os.path.join(val_splits_path, file), "r") as f:
        val_data = json.load(f)
    val_splits_original.append(val_data["validation"])

val_splits_cleansplit = []
for file in cleansplit_files:
    with open(os.path.join(val_splits_path, file), "r") as f:
        val_data = json.load(f)
    val_splits_cleansplit.append(val_data["validation"])

# ------------------------------------------------------------------------------------------------------------------------------
# MLP with Dropout
# ------------------------------------------------------------------------------------------------------------------------------
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        super(MLPModel, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ------------------------------------------------------------------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------------------------------------------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------------------------------------------------------------------------------------------------------
# Early Stopping Class
# ------------------------------------------------------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

# ------------------------------------------------------------------------------------------------------------------------------
# Training function with validation
# ------------------------------------------------------------------------------------------------------------------------------
def train_model_with_validation(X_train, y_train, train_df, val_ids, 
                                 input_dim, device, n_epochs=200, batch_size=32, 
                                 lr=1e-3, hidden_dims=[128, 64], dropout_rate=0.3,
                                 patience=15):
    """
    Train MLP model with custom validation split and early stopping on GPU
    """
    # Create train/val split based on IDs
    id_to_idx = {pid: idx for idx, pid in enumerate(train_df["pdb_id"])}
    val_idx = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
    train_idx = [idx for pid, idx in id_to_idx.items() if pid not in val_ids]
    
    # Split data
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(X_train_fold, y_train_fold)
    val_dataset = EmbeddingDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and move to device
    model = MLPModel(input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * xb.size(0)
        
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                epoch_val_loss += loss.item() * xb.size(0)
        
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping check
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    model.to(device)
    
    return model, train_losses, val_losses

# ------------------------------------------------------------------------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------------------------------------------------------------------------
def evaluate_model(model, X_test_dict, device):
    model.eval()
    results = {}
    predictions = {}
    
    for name, (X_test, y_test) in X_test_dict.items():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy().flatten()
        
        y_true = y_test.values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pcc = pearsonr(y_true, y_pred)[0]
        
        results[name] = {"RMSE": rmse, "R2": r2, "PCC": pcc}
        predictions[name] = (y_true, y_pred)
    
    return results, predictions

# ------------------------------------------------------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------------------------------------------------------
def plot_training_history(train_losses, val_losses, split_name, fold_num, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title(f"MLP - {split_name} Split (Fold {fold_num}) Training History", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"MLP_{split_name}_Fold{fold_num}_TrainingHistory.svg"))
    plt.close()

def plot_predictions(predictions_dict, split_name, fold_num, output_dir):
    n_tests = len(predictions_dict)
    fig, axes = plt.subplots(1, n_tests, figsize=(6*n_tests, 5))
    
    if n_tests == 1:
        axes = [axes]
    
    for idx, (test_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pcc = pearsonr(y_true, y_pred)[0]
        
        ax.set_xlabel("True pK", fontsize=11)
        ax.set_ylabel("Predicted pK", fontsize=11)
        ax.set_title(f"{test_name}\nRMSE={rmse:.3f}, R²={r2:.3f}, PCC={pcc:.3f}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"MLP_{split_name}_Fold{fold_num}_Predictions.svg"))
    plt.close()

# ------------------------------------------------------------------------------------------------------------------------------
# Train MLP with Cross-Validation
# ------------------------------------------------------------------------------------------------------------------------------
splits_config = {
    "Original": {
        "X_train": X_train_orig, 
        "y_train": y_train_orig, 
        "train_df": train_df_orig,
        "val_splits": val_splits_original,
        "test_datasets": test_datasets
    },
    "CleanSplit": {
        "X_train": X_train_clean, 
        "y_train": y_train_clean, 
        "train_df": train_df_clean,
        "val_splits": val_splits_cleansplit,
        "test_datasets": test_datasets_clean
    }
}

# Hyperparameters
HIDDEN_DIMS = [512, 256, 128, 64]
DROPOUT_RATE = 0.3
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
N_EPOCHS = 200
PATIENCE = 20

results_all_folds = {}

for split_name, config in splits_config.items():
    print(f"\n{'='*80}")
    print(f"Training MLP for {split_name} Split with Cross-Validation")
    print(f"{'='*80}")
    
    X_train = config["X_train"]
    y_train = config["y_train"]
    train_df = config["train_df"]
    val_splits = config["val_splits"]
    test_datasets_split = config["test_datasets"]
    
    input_dim = X_train.shape[1]
    
    fold_results = []
    
    # Train on each fold
    for fold_num, val_ids in enumerate(val_splits, 1):
        print(f"\n--- Fold {fold_num}/{len(val_splits)} ---")
        
        model, train_losses, val_losses = train_model_with_validation(
            X_train, y_train, train_df, val_ids,
            input_dim=input_dim,
            device=device,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            hidden_dims=HIDDEN_DIMS,
            dropout_rate=DROPOUT_RATE,
            patience=PATIENCE
        )
        
        # Evaluate on test sets
        results, predictions = evaluate_model(model, test_datasets_split, device)
        fold_results.append(results)
        
        # Plot training history
        plot_training_history(train_losses, val_losses, split_name, fold_num, output_dir)
        
        # Plot predictions
        plot_predictions(predictions, split_name, fold_num, output_dir)
        
        print(f"\nFold {fold_num} Results:")
        for test_name, metrics in results.items():
            print(f"  {test_name}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f}, PCC={metrics['PCC']:.3f}")
    
    results_all_folds[split_name] = fold_results

# ------------------------------------------------------------------------------------------------------------------------------
# Aggregate results across folds
# ------------------------------------------------------------------------------------------------------------------------------
print(f"\n{'='*80}")
print("Cross-Validation Results (Mean ± Std)")
print(f"{'='*80}")

for split_name, fold_results in results_all_folds.items():
    print(f"\n=== {split_name} Split ===")
    
    # Get all test set names
    test_names = fold_results[0].keys()
    
    for test_name in test_names:
        rmse_values = [fold[test_name]["RMSE"] for fold in fold_results]
        r2_values = [fold[test_name]["R2"] for fold in fold_results]
        pcc_values = [fold[test_name]["PCC"] for fold in fold_results]
        
        print(f"\n{test_name}:")
        print(f"  RMSE: {np.mean(rmse_values):.3f} ± {np.std(rmse_values):.3f}")
        print(f"  R²:   {np.mean(r2_values):.3f} ± {np.std(r2_values):.3f}")
        print(f"  PCC:  {np.mean(pcc_values):.3f} ± {np.std(pcc_values):.3f}")

# ------------------------------------------------------------------------------------------------------------------------------
# Save CV results to CSV
# ------------------------------------------------------------------------------------------------------------------------------
cv_results_list = []

for split_name, fold_results in results_all_folds.items():
    test_names = fold_results[0].keys()
    
    for test_name in test_names:
        for fold_num, fold_result in enumerate(fold_results, 1):
            cv_results_list.append({
                'Split': split_name,
                'Test_Set': test_name,
                'Fold': fold_num,
                'RMSE': fold_result[test_name]['RMSE'],
                'R2': fold_result[test_name]['R2'],
                'PCC': fold_result[test_name]['PCC']
            })

cv_results_df = pd.DataFrame(cv_results_list)
csv_path = os.path.join(output_dir, 'MLP_CV_Results.csv')
cv_results_df.to_csv(csv_path, index=False)

print(f"\n{'='*80}")
print("Training completed!")
print(f"- Output directory: {output_dir}")
print(f"- SVG plots saved for each fold")
print(f"- CV results saved to: {csv_path}")
print(f"{'='*80}")