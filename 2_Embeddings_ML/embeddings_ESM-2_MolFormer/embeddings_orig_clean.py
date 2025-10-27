import gc
import json
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error, r2_score

gc.collect()

os.makedirs("plots", exist_ok=True)


def evaluate_model(model, train_data, test_datasets, title, fit=True, figsize=(10, 10), save_path=None):
    X_train, y_train = train_data
    if fit:
        model.fit(X_train, y_train)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for ax, (name, (X_test, y_test)) in zip(axes, test_datasets.items()):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]

        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot(y_test, y_test, color='red', linestyle='--')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_title(f"{name.upper()}\nR²={r2:.3f}, RMSE={rmse:.3f}, PCC={pearson_corr:.3f}")
        ax.set_xlabel("True pK")
        ax.set_ylabel("Predicted pK")
        ax.grid(True, linestyle='--', alpha=0.5)

    if save_path:
            # Entferne jegliche Dateiendung und ersetze sie durch '.svg'
            base, _ = os.path.splitext(save_path)
            final_save_path = base + ".svg"

            # Speichere die Datei explizit als SVG
            plt.savefig(final_save_path, format='svg', bbox_inches='tight')
            plt.close(fig)

def save_cv_results(random_search, filename):
    """
    Speichert die CV-Ergebnisse von RandomizedSearchCV als CSV.
    """
    cv_df = pd.DataFrame(random_search.cv_results_)
    cv_df.to_csv(filename, index=False)

# --- IGNORE ---

df = pd.read_pickle("PDBbind_protein_ligands_embeddings_min_MoLFormer.pkl")

#Concatenate protein and ligand embeddings
df["protein_embedding"] = df["protein_embedding"].apply(
    lambda x: np.concatenate(x) if isinstance(x, (list, np.ndarray)) and np.array(x).ndim > 1 else np.array(x)
)

df["molformer_embedding"] = df["molformer_embedding"].apply(
    lambda x: np.concatenate(x) if isinstance(x, (list, np.ndarray)) and np.array(x).ndim > 1 else np.array(x)
)

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

# Feature-Matrix for Original-Trainingsset
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

# Trainingsdata CleanSplit
train_df_clean = df[df["pdb_id"].isin(train_ids_clean)]
y_train_clean = train_df_clean["pK"]

# Feature-Matrix for CleanSplit-Trainingsset
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

# Map validation IDs to indices for RandomizedSearchCV
def build_cv_splits(train_df, all_val_ids_list, id_col="pdb_id"):
    """
    Converts lists of validation IDs into scikit-learn-compatible
    index tuples (train_idx, val_idx) for RandomizedSearchCV.
    """
    cv_splits = []
    id_to_idx = {pid: idx for idx, pid in enumerate(train_df[id_col])}

    for val_ids in all_val_ids_list:
        val_idx = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
        train_idx = [idx for pid, idx in id_to_idx.items() if pid not in val_ids]
        cv_splits.append((train_idx, val_idx))

    return cv_splits

cv_splits_original = build_cv_splits(train_df_orig, val_splits_original)
cv_splits_cleansplit= build_cv_splits(train_df_clean, val_splits_cleansplit)

# ------------------------------------------------------------------------------
# Linear Regression
model = LinearRegression()

# Predict an evaluate for all test sets
evaluate_model(
    model,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="Linear Regression - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path="plots/lin_orig_split.png"
)
#------- -----------------------------------------------------------------------
# Model
hgb_model = HistGradientBoostingRegressor(
    learning_rate=0.01,
    max_iter=300,
    max_depth=6,
    random_state=42
)
hgb_model.fit(X_train_orig, y_train_orig)

# Predict an evaluate for all test sets
evaluate_model(
    hgb_model,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path="plots/hist_orig_split.png"
)
#------------------------------------------------------------------------------
hgb_model = HistGradientBoostingRegressor(random_state=42)

param_dist = {
    "learning_rate": uniform(0.005, 0.05),   
    "max_iter": randint(100, 600),
    "max_depth": randint(3, 10),
    "min_samples_leaf": randint(10, 80),
    "l2_regularization": uniform(0.0, 1.0),
    "max_bins": randint(100, 255)
}

random_search = RandomizedSearchCV(
    hgb_model,
    param_distributions=param_dist,
    n_iter=20,  
    scoring='r2',
    cv=cv_splits_original, # predefined splits 
    verbose=2, 
    n_jobs=-1,
    random_state=42,
)

random_search.fit(X_train_orig, y_train_orig)
save_cv_results(random_search, "plots/hgb_orig_split_cv_results.csv")

print(random_search.best_params_)
print(f"Best CV R²: {random_search.best_score_:.3f}")

# Best model 
best_hgb = random_search.best_estimator_

# Predict and evaluate for all test sets
evaluate_model(
    best_hgb,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path="plots/hist_orig_split_tuned.png", 
    fit=False
)

# Model with RandomizedSearchCV for CleanSplit
# Parameter grid for RandomizedSearchCV
param_dist = {
    "learning_rate": uniform(0.005, 0.05),
    "max_iter": randint(100, 600),
    "max_depth": randint(3, 10),
    "min_samples_leaf": randint(10, 80),
    "l2_regularization": uniform(0.0, 1.0),
    "max_bins": randint(100, 255)
}

# Model with RandomizedSearchCV

hgb_model = HistGradientBoostingRegressor(
    learning_rate=0.01,
    max_iter=300,
    max_depth=6,
    random_state=42,
    early_stopping=True
)

# Predict an evaluate for all test sets
evaluate_model(
    hgb_model,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path="plots/hist_clean_split.png"
)
# ------------------------------------------------------------------------------
hgb_model = HistGradientBoostingRegressor(random_state=42)

random_search = RandomizedSearchCV(
    hgb_model,
    param_distributions=param_dist,
    n_iter=20,  
    scoring='r2',
    cv=cv_splits_cleansplit, # predefined splits 
    verbose=2, 
    n_jobs=-1,
    random_state=42,
)

random_search.fit(X_train_clean, y_train_clean)
save_cv_results(random_search, "plots/hgb_clean_split_cv_results.csv")

print(random_search.best_params_)
print(f"Best CV R²: {random_search.best_score_:.3f}")

# Best model 
best_hgb = random_search.best_estimator_

# Predict and evaluate for all test sets
evaluate_model(
    best_hgb,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path="plots/hist_clean_split_tuned.png", 
    fit=False
)

# --- XGBoost ---
# Model
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cuda",  
    random_state=42
)

# Predict an evaluate for all test sets
evaluate_model(
    xgb_model,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="XGBoost Regressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path="plots/xgb_orig_split.png"
)
#------------------------------------------------------------------------------ 
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    device='cuda',
    random_state=42
)

params = {
    'n_estimators': randint(300, 1200),          
    'learning_rate': loguniform(1e-3, 0.1),       
    'max_depth': randint(3, 8),                   
    'subsample': uniform(0.7, 0.3),               
    'colsample_bytree': uniform(0.7, 0.3),        
    'reg_alpha': loguniform(1e-3, 10),            
    'reg_lambda': loguniform(0.1, 50)             
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=params,
    n_iter=20,                   
    scoring='r2',                
    cv=cv_splits_original,      # predefined splits 
    verbose=2, 
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train_orig, y_train_orig)
save_cv_results(random_search, "plots/xgb_orig_split_cv_results.csv")

print(random_search.best_params_)
print(f"Best R²: {random_search.best_score_:.4f}")

# Best model
best_model = random_search.best_estimator_

# Predict an evaluate for all test sets
evaluate_model(
    best_model,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path="plots/xgb_orig_split_tuned.png", 
    fit=False
)
#------------------------------------------------------------------------------
# Model
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cpu",  
    random_state=42
)

# Predict an evaluate for all test sets
evaluate_model(
    xgb_model,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="XGBoost Regressor - Performance on CASF Test Sets (Clean Split)",
    save_path="plots/xgb_clean_split.png"
)


# Model with RandomizedSearchCV for CleanSplit
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    device='cuda',
    random_state=42
)

params = {
    'n_estimators': randint(300, 1200),          
    'learning_rate': loguniform(1e-3, 0.1),       
    'max_depth': randint(3, 8),                   
    'subsample': uniform(0.7, 0.3),               
    'colsample_bytree': uniform(0.7, 0.3),        
    'reg_alpha': loguniform(1e-3, 10),            
    'reg_lambda': loguniform(0.1, 50)             
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=params,
    n_iter=20,                   
    scoring='r2',                
    cv=cv_splits_cleansplit,       # predefined splits 
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train_clean, y_train_clean)
save_cv_results(random_search, "plots/xgb_clean_split_cv_results.csv")
print(random_search.best_params_)
print(f"Best R²: {random_search.best_score_:.4f}")

# Best model
best_model = random_search.best_estimator_

# Predict an evaluate for all test sets
evaluate_model(
    best_model,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path="plots/xgb_clean_split_tuned.png", 
    fit=False
)

