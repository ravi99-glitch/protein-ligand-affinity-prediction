import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Scaler for KNN and Linear Models
from sklearn.preprocessing import StandardScaler

# Import all models
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

# Import Hyperparameter distributions
from scipy.stats import uniform, randint, loguniform

# Import Metrics
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------
# Setup: Define save directories
# --------------------------------------------------------------------------
BASE_DIR = "CASF_ML_results"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CV_DIR = os.path.join(BASE_DIR, "cv_results")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")

# Create folders if not existing
for d in [BASE_DIR, PLOTS_DIR, MODELS_DIR, CV_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Results will be stored in: {os.path.abspath(BASE_DIR)}")

# --------------------------------------------------------------------------
# Function to save predictions
# --------------------------------------------------------------------------
def save_predictions(model, test_datasets, pdb_ids_dict, model_prefix):
    """
    Generates and saves prediction JSON files for EACH test dataset separately.
    
    Filename format: {model_prefix}_{dataset_name}_predictions.json
    
    Parameters:
    - model: trained model
    - test_datasets: dict of {name: (X_test, y_test)}
    - pdb_ids_dict: dict of {name: [list_of_ids]}
    - model_prefix: String to prefix filenames (e.g., "ElasticNet_tuned")
    """
    print(f"\n--- Saving Predictions to {PREDICTIONS_DIR} ---")
    
    for name, (X_test, y_test) in test_datasets.items():
        # 1. Check if we have IDs for this dataset
        if name not in pdb_ids_dict:
            print(f"Warning: No PDB IDs found for dataset '{name}'. Skipping save.")
            continue
            
        # 2. Make predictions
        y_pred = model.predict(X_test)
        current_ids = pdb_ids_dict[name]
        
        # 3. Structure the data: {pdb_id: [true, pred]}
        id_to_pred = {}
        for pid, true_val, pred_val in zip(current_ids, y_test, y_pred):
            id_to_pred[pid] = [float(true_val), float(pred_val)]
            
        # 4. Construct dynamic filename
        filename = f"{model_prefix}_{name}_predictions.json"
        save_path = os.path.join(PREDICTIONS_DIR, filename)
        
        # 5. Save to JSON
        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(id_to_pred, json_file, ensure_ascii=False, indent=4)
        
        print(f"Saved: {filename}")

# --------------------------------------------------------------------------
# Function to train and evaluate a regression model with scatter plots
# --------------------------------------------------------------------------
def evaluate_model(model, train_data, test_datasets, title, fit=True, figsize=(10, 10), save_path=None):
    """
    Train and evaluate a regression model on multiple test datasets with scatter plots.
    """

    X_train, y_train = train_data
    if fit:
        model.fit(X_train, y_train)

    num_plots = min(4, len(test_datasets))
    if num_plots == 0:
        print("No test datasets available for evaluation.")
        return

    n_rows = int(np.ceil(num_plots / 2))
    n_cols = min(2, num_plots)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    datasets_to_plot = list(test_datasets.items())[:num_plots]
    
    for ax, (name, (X_test, y_test)) in zip(axes, datasets_to_plot):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # Calculate Pearson Correlation Coefficient (PCC)
        pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]

        ax.scatter(y_test, y_pred, alpha=0.6)
        
        # Plot ideal line (y=x)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--') 
        
        # Consistent plot limits for pK values (typically 0-12)
        plot_min = 0 
        plot_max = 12 
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        ax.set_title(f"{name.upper()}\nR²={r2:.3f}, RMSE={rmse:.3f}, PCC={pearson_corr:.3f}")
        ax.set_xlabel("True pK")
        ax.set_ylabel("Predicted pK")
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Remove unused subplots if necessary
    for i in range(len(datasets_to_plot), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=14, y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="svg", dpi=300)
        print(f"Plot saved as: {save_path}")
    else:
        plt.show()

    plt.close(fig)

def save_cv_results(random_search, filename):
    """
    Speichert die CV-Ergebnisse von RandomizedSearchCV als CSV.
    """
    path = os.path.join(CV_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv_df = pd.DataFrame(random_search.cv_results_)
    cv_df.to_csv(path, index=False)
    print(f"CV results saved to {path}")

# ------------------------------------------------------------------------------------------------
# Data Loading, Splitting, and SCALING
# ------------------------------------------------------------------------------------------------

# Load encoded data and drop non-numeric/sequence columns
drop_cols = ["protein_sequences", "pocket_sequences", "ligand_smiles"]
df_all_encoded = pd.read_csv("PDBbind_protein_pocket_ligands_bindingsites_encoded.csv",)
df_all_encoded = df_all_encoded.drop(columns=drop_cols)
df_all_encoded = df_all_encoded.dropna().reset_index(drop=True)

# Load Original Split IDs
with open("PDBbind_original_data_split.json", "r") as file:
    original_data_split = json.load(file)
train_ids = original_data_split["train"]

# Load Clean Split IDs
with open("PDBbind_cleansplit_data_split.json", "r") as file:
    cleansplit_data_split = json.load(file)
train_ids_clean = cleansplit_data_split["train"]


# --- Data Extraction and Scaling for ORIGINAL Split ---
train_df_orig = df_all_encoded[df_all_encoded["pdb_id"].isin(train_ids)]
X_train_orig = train_df_orig.drop(columns=["pK", "pdb_id"])
y_train_orig = train_df_orig["pK"]

# Initialize and fit Scaler ONLY on original training data
scaler_orig = StandardScaler()
X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig) 

# Prepare original test datasets (scaled only) AND store PDB IDs
test_datasets_scaled = {}
pdb_ids_orig = {}
for key, ids in original_data_split.items():
    if key != 'train': 
        test_df = df_all_encoded[df_all_encoded["pdb_id"].isin(ids)]
        X_test = test_df.drop(columns=["pK", "pdb_id"])
        y_test = test_df["pK"]
        # Scale test data using fitted parameters (only transform!)
        X_test_scaled = scaler_orig.transform(X_test)
        test_datasets_scaled[key] = (X_test_scaled, y_test)
        pdb_ids_orig[key] = test_df["pdb_id"].tolist()


# --- Data Extraction and Scaling for CLEAN Split ---
train_df_clean = df_all_encoded[df_all_encoded["pdb_id"].isin(train_ids_clean)]
X_train_clean = train_df_clean.drop(columns=["pK", "pdb_id"])
y_train_clean = train_df_clean["pK"]

# Initialize and fit Scaler ONLY on clean training data
scaler_clean = StandardScaler()
X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean) 

# Prepare clean test datasets (scaled only) AND store PDB IDs
test_datasets_clean_scaled = {}
pdb_ids_clean = {}
for key, ids in cleansplit_data_split.items():
    if key != 'train': 
        test_df = df_all_encoded[df_all_encoded["pdb_id"].isin(ids)]
        X_test = test_df.drop(columns=["pK", "pdb_id"])
        y_test = test_df["pK"]
        # Scale test data using fitted parameters (only transform!)
        X_test_scaled = scaler_clean.transform(X_test)
        test_datasets_clean_scaled[key] = (X_test_scaled, y_test)
        pdb_ids_clean[key] = test_df["pdb_id"].tolist()

# ------------------------------------------------------------------------------------------------
# Cross-Validation Data Setup (Splits preparation)
# ------------------------------------------------------------------------------------------------

val_splits_path = "val_splits"
val_split_files = [f for f in os.listdir(val_splits_path) if f.endswith(".json")]

original_files = sorted([f for f in val_split_files if f.startswith("original")])
cleansplit_files = sorted([f for f in val_split_files if f.startswith("cleansplit")])

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
cv_splits_cleansplit = build_cv_splits(train_df_clean, val_splits_cleansplit)

print(f"Original CV folds: {len(cv_splits_original)}")
print(f"CleanSplit CV folds: {len(cv_splits_cleansplit)}")
# ------------------------------------------------------------------------------------------------
# Define KNN parameter grid once
param_dist_knn = {
    'n_neighbors': randint(3, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': randint(1, 5) 
}

# Define ElasticNet parameter grid once
param_dist_en = {
    'alpha': loguniform(0.001, 100.0),
    'l1_ratio': uniform(0.0, 1.0)
}

# Define HGB parameter grid once
param_dist_hgb = {
    "learning_rate": uniform(0.005, 0.05),  
    "max_iter": randint(100, 600),
    "max_depth": randint(3, 10),
    "min_samples_leaf": randint(10, 80),
    "l2_regularization": uniform(0.0, 1.0),
    "max_bins": randint(100, 255)
}

# Define XGBoost parameter grid once
params_xgb = {
    'n_estimators': randint(300, 1200), 
    'learning_rate': loguniform(1e-3, 0.1), 
    'max_depth': randint(3, 8), 
    'subsample': uniform(0.7, 0.3), 
    'colsample_bytree': uniform(0.7, 0.3), 
    'reg_alpha': loguniform(1e-3, 10), 
    'reg_lambda': loguniform(0.1, 50) 
}

# ================================================================================================
# 1. K-NEAREST NEIGHBORS (KNN)
# ================================================================================================

## 1.1 KNN Base Model (Original Split)
model_orig_base = KNeighborsRegressor(n_neighbors=5) 
evaluate_model(
    model_orig_base,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="KNN Regression - Performance on CASF Test Sets (Original Split - Base)",
    save_path = os.path.join(PLOTS_DIR, "knn_base_orig.svg")
)

## 1.2 KNN Hyperparameter Tuning (Original Split)
random_search_knn_orig = RandomizedSearchCV(
    estimator=KNeighborsRegressor(),
    param_distributions=param_dist_knn,
    n_iter=20, 
    scoring='r2', 
    cv=cv_splits_original, 
    verbose=2, 
    n_jobs=-1, 
    random_state=42
)
random_search_knn_orig.fit(X_train_orig_scaled, y_train_orig) 

cv_results_df_knn_orig = pd.DataFrame(random_search_knn_orig.cv_results_)
csv_path_knn_orig = os.path.join(CV_DIR, "knn_tuned_cv_original.csv")
cv_results_df_knn_orig.to_csv(csv_path_knn_orig, index=False)
print(f"CV results saved as: {csv_path_knn_orig}")

print("\n--- Best KNN Params (Original Split) ---")
print(random_search_knn_orig.best_params_)
print(f"Best CV R²: {random_search_knn_orig.best_score_:.3f}")
best_knn_orig = random_search_knn_orig.best_estimator_ 

# Save predictions
save_predictions(best_knn_orig, test_datasets_scaled, pdb_ids_orig, "KNN_tuned_orig")

evaluate_model(
    best_knn_orig,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="Tuned KNN - Performance on CASF Test Sets (Original Split)",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "knn_tuned_orig.svg")
)

# ------------------------------------------------------------------------------------------------

# ================================================================================================
# 2. ELASTIC NET - ORIGINAL SPLIT 
# ================================================================================================

# Base Model
model_en = ElasticNet(random_state=42, max_iter=5000)
evaluate_model(
    model_en,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="ElasticNet - Performance on CASF Test Sets (Original Split)",
    save_path = os.path.join(PLOTS_DIR, "elasticnet.svg")
)

# Tuning
random_search_en = RandomizedSearchCV(
    estimator=model_en,
    param_distributions=param_dist_en,
    n_iter=20, 
    scoring='r2', 
    cv=cv_splits_original, 
    verbose=2, 
    n_jobs=-1, 
    random_state=42,  
)

random_search_en.fit(X_train_orig_scaled, y_train_orig) 

cv_results_df_en = pd.DataFrame(random_search_en.cv_results_)
csv_path_en = os.path.join(CV_DIR, "elasticnet_tuned_cv.csv")
cv_results_df_en.to_csv(csv_path_en, index=False)
print(f"CV results saved as: {csv_path_en}")

print(random_search_en.best_params_)
print(f"Best CV R²: {random_search_en.best_score_:.3f}")

best_elasticnet = random_search_en.best_estimator_

# Save predictions
save_predictions(best_elasticnet, test_datasets_scaled, pdb_ids_orig, "ElasticNet_tuned_orig")

evaluate_model(
    best_elasticnet,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="Tuned ElasticNet - Performance on CASF Test Sets",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "elasticnet_tuned.svg")
)

# ------------------------------------------------------------------------------------------------

# ================================================================================================
# 3. HIST GRADIENT BOOSTING (HGB) - ORIGINAL SPLIT 
# ================================================================================================

# Base Model
hgb_model_base = HistGradientBoostingRegressor(
    learning_rate=0.01,
    max_iter=300,
    max_depth=6,
    random_state=42
)

hgb_model_base.fit(X_train_orig_scaled, y_train_orig) 

evaluate_model(
    hgb_model_base,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled,
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Original Split)",
    save_path = os.path.join(PLOTS_DIR, "hgb.svg")
)

# Tuning
random_search_hgb = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42),
    param_distributions=param_dist_hgb,
    n_iter=20,  
    scoring='r2',
    cv=cv_splits_original, 
    verbose=2, 
    n_jobs=-1,
    random_state=42,
)

random_search_hgb.fit(X_train_orig_scaled, y_train_orig) 

cv_results_df_hgb = pd.DataFrame(random_search_hgb.cv_results_)
csv_path_hgb = os.path.join(CV_DIR, "hgb_tuned_cv.csv")
cv_results_df_hgb.to_csv(csv_path_hgb, index=False)
print(f"CV results saved as: {csv_path_hgb}")

print(random_search_hgb.best_params_)
print(f"Best CV R²: {random_search_hgb.best_score_:.3f}")

best_hgb = random_search_hgb.best_estimator_

# Save predictions
save_predictions(best_hgb, test_datasets_scaled, pdb_ids_orig, "HGB_tuned_orig")

evaluate_model(
    best_hgb,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "hgb_tuned.svg")
)

# ------------------------------------------------------------------------------------------------

# ================================================================================================
# 4. XGBOOST - ORIGINAL SPLIT
# ================================================================================================

# Base Model
xgb_model_base = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cuda",  
    random_state=42
)
xgb_model_base.fit(X_train_orig_scaled, y_train_orig) 

evaluate_model(
    xgb_model_base,
    (X_train_orig_scaled, y_train_orig),
    test_datasets_scaled, 
    title="XGBoost Regressor - Performance on CASF Test Sets (Original Split)",
    save_path = os.path.join(PLOTS_DIR,"xgb.svg")
)

# Tuning
xgb_model_tuned_orig = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    device='cuda',
    random_state=42
)

random_search_xgb_orig = RandomizedSearchCV(
    estimator=xgb_model_tuned_orig,
    param_distributions=params_xgb,
    n_iter=20, 
    scoring='r2', 
    cv=cv_splits_original, 
    verbose=2, 
    random_state=42,
    n_jobs=-1,
)

random_search_xgb_orig.fit(X_train_orig_scaled, y_train_orig) 

cv_results_df_xgb_orig = pd.DataFrame(random_search_xgb_orig.cv_results_)
csv_path_xgb_orig = os.path.join(CV_DIR, "xgb_tuned_cv.csv")
cv_results_df_xgb_orig.to_csv(csv_path_xgb_orig, index=False)
print(f"CV results saved as: {csv_path_xgb_orig}")

print(random_search_xgb_orig.best_params_)
print(f"Best R²: {random_search_xgb_orig.best_score_:.4f}")

best_xgb_orig = random_search_xgb_orig.best_estimator_

# Save predictions
save_predictions(best_xgb_orig, test_datasets_scaled, pdb_ids_orig, "XGB_tuned_orig")

evaluate_model(
    best_xgb_orig,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Original Split)", 
    fit=False,
    save_path = os.path.join(PLOTS_DIR,"xgb_tuned.svg")
)

# ------------------------------------------------------------------------------------------------
# ================================================================================================
# 5. K-NEAREST NEIGHBORS (KNN) - CLEAN SPLIT 
# ================================================================================================

## 5.1 KNN Base Model (Clean Split)
model_clean_base = KNeighborsRegressor(n_neighbors=5)
evaluate_model(
    model_clean_base,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled,
    title="KNN Regression - Performance on CASF Test Sets (Clean Split - Base)",
    save_path = os.path.join(PLOTS_DIR, "knn_base_clean.svg")
)

## 5.2 KNN Hyperparameter Tuning (Clean Split)
random_search_knn_clean = RandomizedSearchCV(
    estimator=KNeighborsRegressor(),
    param_distributions=param_dist_knn,
    n_iter=20, 
    scoring='r2', 
    cv=cv_splits_cleansplit, 
    verbose=2, 
    n_jobs=-1, 
    random_state=42
)
random_search_knn_clean.fit(X_train_clean_scaled, y_train_clean) 

cv_results_df_knn_clean = pd.DataFrame(random_search_knn_clean.cv_results_)
csv_path_knn_clean = os.path.join(CV_DIR, "knn_tuned_cv_clean.csv")
cv_results_df_knn_clean.to_csv(csv_path_knn_clean, index=False)
print(f"CV results saved as: {csv_path_knn_clean}")

print("\n--- Best KNN Params (Clean Split) ---")
print(random_search_knn_clean.best_params_)
print(f"Best CV R²: {random_search_knn_clean.best_score_:.3f}")

best_knn_clean = random_search_knn_clean.best_estimator_ 

# Save predictions
save_predictions(best_knn_clean, test_datasets_clean_scaled, pdb_ids_clean, "KNN_tuned_clean")

evaluate_model(
    best_knn_clean,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="Tuned KNN - Performance on CASF Test Sets (Clean Split)",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "knn_tuned_clean.svg")
)

# ------------------------------------------------------------------------------------------------

# ================================================================================================
# 6. ELASTIC NET - CLEAN SPLIT 
# ================================================================================================

# Base Model
model_clean_en = ElasticNet(random_state=42, max_iter=5000)

evaluate_model(
    model_clean_en,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="ElasticNet - Performance on CASF Test Sets (Clean Split)",
    save_path = os.path.join(PLOTS_DIR, "elasticnet_clean.svg")
)

# Tuning
random_search_en_clean = RandomizedSearchCV(
    estimator=model_clean_en,
    param_distributions=param_dist_en,
    n_iter=20, 
    scoring='r2', 
    cv=cv_splits_cleansplit, 
    verbose=2, 
    n_jobs=-1, 
    random_state=42,  
)
random_search_en_clean.fit(X_train_clean_scaled, y_train_clean) 

cv_results_df_en_clean = pd.DataFrame(random_search_en_clean.cv_results_)
csv_path_en_clean = os.path.join(CV_DIR, "elasticnet_tuned_cv_clean.csv")
cv_results_df_en_clean.to_csv(csv_path_en_clean, index=False)
print(f"CV results saved as: {csv_path_en_clean}")

print(random_search_en_clean.best_params_)
print(f"Best CV R²: {random_search_en_clean.best_score_:.3f}")

best_elasticnet_clean = random_search_en_clean.best_estimator_

# Save predictions
save_predictions(best_elasticnet_clean, test_datasets_clean_scaled, pdb_ids_clean, "ElasticNet_tuned_clean")

evaluate_model(
    best_elasticnet_clean,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="Tuned ElasticNet - Performance on CASF Test Sets (Clean Split)",
    fit=False,
    save_path = os.path.join(PLOTS_DIR, "elasticnet_tuned_clean.svg")
)

# ------------------------------------------------------------------------------------------------

# ================================================================================================
# 7. HIST GRADIENT BOOSTING (HGB) - CLEAN SPLIT 
# ================================================================================================

# Base Model
hgb_model_clean_base = HistGradientBoostingRegressor(
    learning_rate=0.01,
    max_iter=300,
    max_depth=6,
    random_state=42,
    early_stopping=True
)
hgb_model_clean_base.fit(X_train_clean_scaled, y_train_clean) 

evaluate_model(
    hgb_model_clean_base,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Clean Split)",
    save_path = os.path.join(PLOTS_DIR,"hgb_clean.svg")
)

# Tuning
hgb_model_tuned_clean = HistGradientBoostingRegressor(random_state=42)

random_search_hgb_clean = RandomizedSearchCV(
    hgb_model_tuned_clean,
    param_distributions=param_dist_hgb, 
    n_iter=20,  
    scoring='r2',
    cv=cv_splits_cleansplit, 
    verbose=2, 
    n_jobs=-1,
    random_state=42,
)

random_search_hgb_clean.fit(X_train_clean_scaled, y_train_clean) 

cv_results_df_hgb_clean = pd.DataFrame(random_search_hgb_clean.cv_results_)
csv_path_hgb_clean = os.path.join(CV_DIR, "hgb_tuned_cv_clean.csv")
cv_results_df_hgb_clean.to_csv(csv_path_hgb_clean, index=False)
print(f"CV results saved as: {csv_path_hgb_clean}")

print(random_search_hgb_clean.best_params_)
print(f"Best CV R²: {random_search_hgb_clean.best_score_:.3f}")

best_hgb_clean = random_search_hgb_clean.best_estimator_

# Save predictions
save_predictions(best_hgb_clean, test_datasets_clean_scaled, pdb_ids_clean, "HGB_tuned_clean")

evaluate_model(
    best_hgb_clean,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets (Clean Split)", 
    fit=False,
    save_path = os.path.join(PLOTS_DIR,"hgb_tuned_clean.svg")
)

# ------------------------------------------------------------------------------------------------

# ================================================================================================
# 8. XGBOOST - CLEAN SPLIT 
# ================================================================================================

# Base Model
xgb_model_clean_base = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cpu", 
    random_state=42
)
xgb_model_clean_base.fit(X_train_clean_scaled, y_train_clean) 

evaluate_model(
    xgb_model_clean_base,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="XGBoost Regressor - Performance on CASF Test Sets (Clean Split)",
    save_path = os.path.join(PLOTS_DIR,"xgb_clean.svg")
)

# Tuning
xgb_model_tuned_clean = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    device='cuda',
    random_state=42
)

random_search_xgb_clean = RandomizedSearchCV(
    estimator=xgb_model_tuned_clean,
    param_distributions=params_xgb, 
    n_iter=20, 
    scoring='r2', 
    cv=cv_splits_cleansplit, 
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

random_search_xgb_clean.fit(X_train_clean_scaled, y_train_clean) 

cv_results_df_xgb_clean = pd.DataFrame(random_search_xgb_clean.cv_results_)
csv_path_xgb_clean = os.path.join(CV_DIR, "xgb_tuned_cv_clean.csv")
cv_results_df_xgb_clean.to_csv(csv_path_xgb_clean, index=False)
print(f"CV results saved as: {csv_path_xgb_clean}")

print(random_search_xgb_clean.best_params_)
print(f"Best R²: {random_search_xgb_clean.best_score_:.4f}")

best_xgb_clean = random_search_xgb_clean.best_estimator_

# Save predictions
save_predictions(best_xgb_clean, test_datasets_clean_scaled, pdb_ids_clean, "XGB_tuned_clean")

evaluate_model(
    best_xgb_clean,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Clean Split)", 
    fit=False,
    save_path = os.path.join(PLOTS_DIR,"xgb_tuned_clean.svg")
)

print("\n" + "="*80)
print("ALL MODELS TRAINED AND PREDICTIONS SAVED!")
print("="*80)