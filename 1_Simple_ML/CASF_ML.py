import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------
# Setup: Define save directories
# --------------------------------------------------------------------------
BASE_DIR = "CASF_ML_results"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CV_DIR = os.path.join(BASE_DIR, "cv_results")

# Create folders if not existing
for d in [BASE_DIR, PLOTS_DIR, MODELS_DIR, CV_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Results will be stored in: {os.path.abspath(BASE_DIR)}")
# --------------------------------------------------------------------------
# Function to train and evaluate a regression model with scatter plots
def evaluate_model(model, train_data, test_datasets, title, fit=True, figsize=(10, 10), save_path=None):
    """
    Train and evaluate a regression model on multiple test datasets with scatter plots.
    
    Parameters
    ----------
    model : sklearn-like regressor
        The model to train and evaluate.
    train_data : tuple
        (X_train, y_train) data for fitting if fit=True.
    test_datasets : dict
        Dictionary of test datasets: {"name": (X_test, y_test), ...}.
    title : str
        Title for the entire figure.
    fit : bool, optional
        Whether to fit the model before evaluating (default: True).
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        If provided, saves the plot to this path (e.g. "plots/results.svg").
    """

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

    fig.suptitle(title, fontsize=14, y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="svg", dpi=300)
        print(f"Plot saved as: {save_path}")
    else:
        plt.show()

    plt.close(fig)

# ------------------------------------------------------------------------------------------------
# Train - Test Split

# Load encoded data without pocket sequences
drop_cols = ["protein_sequences", "pocket_sequences", "ligand_smiles"]

df_all_encoded = pd.read_csv(
    "PDBbind_protein_pocket_ligands_bindingsites_encoded.csv",)
df_all_encoded = df_all_encoded.drop(columns=drop_cols)
df_all_encoded = df_all_encoded.dropna().reset_index(drop=True)

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

print("Original split keys:", original_data_split.keys())
print("Clean split keys:", cleansplit_data_split.keys())

# Trainingsdata original
train_df_orig = df_all_encoded[df_all_encoded["pdb_id"].isin(train_ids)]
X_train_orig = train_df_orig.drop(columns=["pK", "pdb_id"])
y_train_orig = train_df_orig["pK"]

# Testdata original 
test_datasets = {}
for key, ids in original_data_split.items():
    if key != 'train':  
        test_df = df_all_encoded[df_all_encoded["pdb_id"].isin(ids)]
        X_test = test_df.drop(columns=["pK", "pdb_id"])
        y_test = test_df["pK"]
        test_datasets[key] = (X_test, y_test)

X_test_2013, y_test_2013 = test_datasets["casf2013"]
X_test_2016, y_test_2016 = test_datasets["casf2016"]
X_test_2013_indep, y_test_2013_indep = test_datasets["casf2013_indep"]
X_test_2016_indep, y_test_2016_indep = test_datasets["casf2016_indep"]

# Trainingsdata clean
train_df_clean = df_all_encoded[df_all_encoded["pdb_id"].isin(train_ids_clean)]
X_train_clean = train_df_clean.drop(columns=["pK", "pdb_id"])
y_train_clean = train_df_clean["pK"]

# Testdata clean
test_datasets_clean = {}
for key, ids in cleansplit_data_split.items():
    if key != 'train': 
        test_df = df_all_encoded[df_all_encoded["pdb_id"].isin(ids)]
        X_test = test_df.drop(columns=["pK", "pdb_id"])
        y_test = test_df["pK"]
        test_datasets_clean[key] = (X_test, y_test)

X_test_clean_casf2013, y_test_clean_casf2013 = test_datasets_clean.get("casf2013", (None, None))
X_test_clean_casf2016, y_test_clean_casf2016 = test_datasets_clean.get("casf2016", (None, None))
X_test_clean_casf2013_indep, y_test_clean_casf2013_indep = test_datasets_clean.get("casf2013_indep", (None, None))
X_test_clean_casf2016_indep, y_test_clean_casf2016_indep = test_datasets_clean.get("casf2016_indep", (None, None))
# ------------------------------------------------------------------------------------------------
#Cross-Validation Data with specific train-validation splits

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

print(f"Original CV folds: {len(cv_splits_original)}")
print(f"CleanSplit CV folds: {len(cv_splits_cleansplit)}")
# ------------------------------------------------------------------------------------------------
# Model
model = ElasticNet(random_state=42, max_iter=10000)

# Predict an evaluate for all test sets
evaluate_model(
    model,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="ElasticNet - Performance on CASF Test Sets (Original Split)",
    save_path = os.path.join(PLOTS_DIR, "elasticnet.svg")
)
# ------------------------------------------------------------------------------------------------
# Paramter grid for RandomizedSearchCV
param_dist = {
    'alpha': uniform(0.001, 100.0),
    'l1_ratio': uniform(0.0, 1.0)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,                  
    scoring='r2',               
    cv=cv_splits_original,      # predefined splits
    verbose=2,                  
    n_jobs=-1,                  
    random_state=42,  
)

random_search.fit(X_train_orig, y_train_orig)

# Save CV results to CSV
cv_results_df = pd.DataFrame(random_search.cv_results_)
csv_path = os.path.join(CV_DIR, "elasticnet_tuned_cv.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cv_results_df.to_csv(csv_path, index=False)
print(f"CV results saved as: {csv_path}")

print(random_search.best_params_)
print(f"Best CV R²: {random_search.best_score_:.3f}")

# Best model 
best_elasticnet = random_search.best_estimator_

# Predict and evaluate for all test sets
evaluate_model(
    best_elasticnet,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="Tuned ElasticNet - Performance on CASF Test Sets",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "elasticnet_tuned.svg")
)
# ------------------------------------------------------------------------------------------------
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
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Original Split)",
    save_path = os.path.join(PLOTS_DIR, "hgb.svg")
)
# ------------------------------------------------------------------------------------------------
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
hgb_model = HistGradientBoostingRegressor(random_state=42)

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

# Save CV results to CSV
cv_results_df = pd.DataFrame(random_search.cv_results_)
csv_path = os.path.join(CV_DIR, "hgb_tuned_cv.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cv_results_df.to_csv(csv_path, index=False)
print(f"CV results saved as: {csv_path}")

print(random_search.best_params_)
print(f"Best CV R²: {random_search.best_score_:.3f}")

# Best model 
best_hgb = random_search.best_estimator_

# Predict and evaluate for all test sets
evaluate_model(
    best_hgb,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "hgb_tuned.svg")
)

# ------------------------------------------------------------------------------------------------
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
    title="XGBoost Regressor - Performance on CASF Test Sets (Original Split)",
    save_path = os.path.join(PLOTS_DIR,"xgb.svg")
)


# ------------------------------------------------------------------------------------------------
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

# Save CV results to CSV
cv_results_df = pd.DataFrame(random_search.cv_results_)
csv_path = os.path.join(CV_DIR, "xgb_tuned_cv.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cv_results_df.to_csv(csv_path, index=False)
print(f"CV results saved as: {csv_path}")

print(random_search.best_params_)
print(f"Best R²: {random_search.best_score_:.4f}")

# Best model
best_model = random_search.best_estimator_

# Predict an evaluate for all test sets
evaluate_model(
    best_model,
    (X_train_orig, y_train_orig),
    test_datasets,
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Original Split)", 
    fit=False,
    save_path = os.path.join(PLOTS_DIR,"xgb_tuned.svg")
)


# ------------------------------------------------------------------------------------------------
# Model
model_clean = ElasticNet(random_state=42, max_iter=10000)

# Predict an evaluate for all test sets
evaluate_model(
    model_clean,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="ElasticNet - Performance on CASF Test Sets (Clean Split)",
    save_path = os.path.join(PLOTS_DIR, "elasticnet_clean.svg")
)
# ------------------------------------------------------------------------------------------------
# Paramter grid for RandomizedSearchCV
param_dist = {
    'alpha': uniform(0.001, 100.0),
    'l1_ratio': uniform(0.0, 1.0)
}

random_search = RandomizedSearchCV(
    estimator=model_clean,
    param_distributions=param_dist,
    n_iter=20,                  
    scoring='r2',               
    cv=cv_splits_cleansplit,      # predefined splits
    verbose=2,                  
    n_jobs=-1,                  
    random_state=42,  
)

random_search.fit(X_train_clean, y_train_clean)

# Save CV results to CSV
cv_results_df = pd.DataFrame(random_search.cv_results_)
csv_path = os.path.join(CV_DIR, "elasticnet_tuned_cv_clean.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cv_results_df.to_csv(csv_path, index=False)
print(f"CV results saved as: {csv_path}")

print(random_search.best_params_)
print(f"Best CV R²: {random_search.best_score_:.3f}")

# Best model 
best_elasticnet = random_search.best_estimator_

# Predict and evaluate for all test sets
evaluate_model(
    best_elasticnet,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="Tuned ElasticNet - Performance on CASF Test Sets (Clean Split)",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "elasticnet_tuned_clean.svg")
)

# ------------------------------------------------------------------------------------------------

# Model
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
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Clean Split)",
    save_path = os.path.join(PLOTS_DIR,"hgb_clean.svg")
)

# ------------------------------------------------------------------------------------------------
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

# Save CV results to CSV
cv_results_df = pd.DataFrame(random_search.cv_results_)
csv_path = os.path.join(CV_DIR, "hgb_tuned_cv_clean.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cv_results_df.to_csv(csv_path, index=False)
print(f"CV results saved as: {csv_path}")


print(random_search.best_params_)
print(f"Best CV R²: {random_search.best_score_:.3f}")

# Best model 
best_hgb = random_search.best_estimator_

# Predict and evaluate for all test sets
evaluate_model(
    best_hgb,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets (Clean Split)", 
    fit=False,
    save_path = os.path.join(PLOTS_DIR,"hgb_tuned_clean.svg")
)

# ------------------------------------------------------------------------------------------------
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
    save_path = os.path.join(PLOTS_DIR,"xgb_clean.svg")
)
# ------------------------------------------------------------------------------------------------
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

# Save CV results to CSV
cv_results_df = pd.DataFrame(random_search.cv_results_)
csv_path = os.path.join(CV_DIR, "xgb_tuned_cv_clean.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
cv_results_df.to_csv(csv_path, index=False)
print(f"CV results saved as: {csv_path}")


print(random_search.best_params_)
print(f"Best R²: {random_search.best_score_:.4f}")

# Best model
best_model = random_search.best_estimator_

# Predict an evaluate for all test sets
evaluate_model(
    best_model,
    (X_train_clean, y_train_clean),
    test_datasets_clean,
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Clean Split)", 
    fit=False,
    save_path = os.path.join(PLOTS_DIR,"xgb_tuned_clean.svg")
)
