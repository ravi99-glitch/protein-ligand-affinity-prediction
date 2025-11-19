import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor 
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------------------------------
# Setup: Define save directories
# --------------------------------------------------------------------------
BASE_DIR = "CASF_embeddings_ML_results"
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
        
        print(f" -> Saved: {filename}")

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
    Saves the cross-validation results from a RandomizedSearchCV object to a CSV file
    """
    path = os.path.join(CV_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv_df = pd.DataFrame(random_search.cv_results_)
    cv_df.to_csv(path, index=False)
    print(f"CV results saved to {path}")


# ------------------------------------------------------------------------------
# --- Data Loading and Processing ---

df = pd.read_pickle("PDBbind_protein_ligands_embeddings_min_MoLFormer.pkl")

# Concatenate protein and ligand embeddings
df["embedding"] = df.apply(
    lambda row: np.concatenate([
        np.concatenate(row["protein_embedding"]) if isinstance(row["protein_embedding"], (list, np.ndarray)) and np.array(row["protein_embedding"]).ndim > 1 else np.array(row["protein_embedding"]),
        np.concatenate(row["molformer_embedding"]) if isinstance(row["molformer_embedding"], (list, np.ndarray)) and np.array(row["molformer_embedding"]).ndim > 1 else np.array(row["molformer_embedding"])
    ]), 
    axis=1
)

# Load PDBbind splits
with open("PDBbind_original_data_split.json", "r") as file:
    original_data_split = json.load(file)
train_ids_orig = original_data_split["train"]

with open("PDBbind_cleansplit_data_split.json", "r") as file:
    cleansplit_data_split = json.load(file)
train_ids_clean = cleansplit_data_split["train"]


# --- Original Split Data Prep ---
train_df_orig = df[df["pdb_id"].isin(train_ids_orig)]
y_train_orig = train_df_orig["pK"].values
X_train_orig_unscaled = np.vstack(train_df_orig["embedding"])

# --- Scaling for Original Split ---
scaler_orig = StandardScaler()
X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig_unscaled)

# Testdata original AND store PDB IDs
test_datasets_scaled = {}
pdb_ids_orig = {}
for key, ids in original_data_split.items():
    if key != 'train': 
        test_df = df[df["pdb_id"].isin(ids)]
        y_test = test_df["pK"].values
        X_test_unscaled = np.vstack(test_df["embedding"])
        
        # Scaling test data
        X_test_scaled = scaler_orig.transform(X_test_unscaled)
        test_datasets_scaled[key] = (X_test_scaled, y_test)
        pdb_ids_orig[key] = test_df["pdb_id"].tolist()


# --- Clean Split Data Prep ---
train_df_clean = df[df["pdb_id"].isin(train_ids_clean)]
y_train_clean = train_df_clean["pK"].values
X_train_clean_unscaled = np.vstack(train_df_clean["embedding"])

# --- Scaling for Clean Split ---
scaler_clean = StandardScaler()
X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean_unscaled)

# Testdata CleanSplit AND store PDB IDs
test_datasets_clean_scaled = {}
pdb_ids_clean = {}
for key, ids in cleansplit_data_split.items():
    if key != 'train': 
        test_df = df[df["pdb_id"].isin(ids)]
        y_test = test_df["pK"].values
        X_test_unscaled = np.vstack(test_df["embedding"])
        
        # Scaling test data
        X_test_scaled = scaler_clean.transform(X_test_unscaled)
        test_datasets_clean_scaled[key] = (X_test_scaled, y_test)
        pdb_ids_clean[key] = test_df["pdb_id"].tolist()

# --- CV Splits ---
def build_cv_splits(train_df, all_val_ids_list, id_col="pdb_id"):
    cv_splits = []
    id_to_idx = {pid: idx for idx, pid in enumerate(train_df[id_col])}
    for val_ids in all_val_ids_list:
        val_idx = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
        train_idx = [idx for pid, idx in id_to_idx.items() if pid not in val_ids]
        cv_splits.append((train_idx, val_idx))
    return cv_splits

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

cv_splits_orig = build_cv_splits(train_df_orig, val_splits_original)
cv_splits_clean = build_cv_splits(train_df_clean, val_splits_cleansplit)

# ------------------------------------------------------------------------------
# --- Parameter Grids ---
param_dist_elasticnet = {
    'alpha': loguniform(0.001, 100.0), 
    'l1_ratio': uniform(0.0, 1.0) 
}
param_dist_hgb = {
    "learning_rate": uniform(0.005, 0.05),  
    "max_iter": randint(100, 600),
    "max_depth": randint(3, 10),
    "min_samples_leaf": randint(10, 80),
    "l2_regularization": uniform(0.0, 1.0),
    "max_bins": randint(100, 255)
}
param_dist_xgb = {
    'n_estimators': randint(300, 1200), 
    'learning_rate': loguniform(1e-3, 0.1), 
    'max_depth': randint(3, 8), 
    'subsample': uniform(0.7, 0.3), 
    'colsample_bytree': uniform(0.7, 0.3), 
    'reg_alpha': loguniform(1e-3, 10), 
    'reg_lambda': loguniform(0.1, 50) 
}
param_dist_knn = {
    'n_neighbors': randint(3, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': randint(1, 5) 
}
# ------------------------------------------------------------------------------
# ==============================================================================
# 1. K-NEAREST NEIGHBORS (KNN)
# ==============================================================================

# KNN Baseline - Original Split
knn_model_orig = KNeighborsRegressor(n_neighbors=5)
evaluate_model(
    knn_model_orig,
    (X_train_orig_scaled, y_train_orig),
    test_datasets_scaled,
    title="KNN - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/knn_orig_split.svg"
)

# KNN - RandomizedSearchCV - Original Split
random_search_knn_orig = RandomizedSearchCV(
    estimator=KNeighborsRegressor(),
    param_distributions=param_dist_knn,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_orig,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)

random_search_knn_orig.fit(X_train_orig_scaled, y_train_orig)
save_cv_results(random_search_knn_orig, f"knn_orig_split_cv_results.csv")

print("KNN Original Split - Best params:", random_search_knn_orig.best_params_)
print(f"KNN Original Split - Best CV R²: {random_search_knn_orig.best_score_:.3f}")

best_knn_orig = random_search_knn_orig.best_estimator_

# Save predictions
save_predictions(best_knn_orig, test_datasets_scaled, pdb_ids_orig, "KNN_tuned_orig")

evaluate_model(
    best_knn_orig,
    (X_train_orig_scaled, y_train_orig),
    test_datasets_scaled,
    title="Tuned KNN - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/knn_orig_split_tuned.svg", 
    fit=False
)

# KNN Baseline - CleanSplit
knn_model_clean = KNeighborsRegressor(n_neighbors=5)
evaluate_model(
    knn_model_clean,
    (X_train_clean_scaled, y_train_clean),
    test_datasets_clean_scaled,
    title="KNN - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path=f"{PLOTS_DIR}/knn_clean_split.svg"
)

# KNN - RandomizedSearchCV - CleanSplit
random_search_knn_clean = RandomizedSearchCV(
    estimator=KNeighborsRegressor(),
    param_distributions=param_dist_knn,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_clean,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)

random_search_knn_clean.fit(X_train_clean_scaled, y_train_clean)
save_cv_results(random_search_knn_clean, f"knn_clean_split_cv_results.csv")

print("KNN Clean Split - Best params:", random_search_knn_clean.best_params_)
print(f"KNN Clean Split - Best CV R²: {random_search_knn_clean.best_score_:.3f}")

best_knn_clean = random_search_knn_clean.best_estimator_

# Save predictions
save_predictions(best_knn_clean, test_datasets_clean_scaled, pdb_ids_clean, "KNN_tuned_clean")

evaluate_model(
    best_knn_clean,
    (X_train_clean_scaled, y_train_clean),
    test_datasets_clean_scaled,
    title="Tuned KNN - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path=f"{PLOTS_DIR}/knn_clean_split_tuned.svg", 
    fit=False
)

# ==============================================================================
# 2. ELASTIC NET
# ==============================================================================

# ElasticNet Baseline - Original Split 
model = ElasticNet(random_state=42)

evaluate_model(
    model,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="ElasticNet - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/elasticnet_orig_split.svg"
)

# ElasticNet Baseline - CleanSplit
model_clean = ElasticNet(random_state=42)

evaluate_model(
    model_clean,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="ElasticNet - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path=f"{PLOTS_DIR}/elasticnet_clean_split.svg"
)

# ElasticNet - RandomizedSearchCV - Original Split
random_search_orig = RandomizedSearchCV(
    estimator=ElasticNet(random_state=42),
    param_distributions=param_dist_elasticnet,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_orig,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)

random_search_orig.fit(X_train_orig_scaled, y_train_orig)
save_cv_results(random_search_orig, f"elasticnet_orig_split_cv_results.csv")

print("ElasticNet Original Split - Best params:", random_search_orig.best_params_)
print(f"ElasticNet Original Split - Best CV R²: {random_search_orig.best_score_:.3f}")

best_elasticnet_orig = random_search_orig.best_estimator_

# Save predictions
save_predictions(best_elasticnet_orig, test_datasets_scaled, pdb_ids_orig, "ElasticNet_tuned_orig")

evaluate_model(
    best_elasticnet_orig,
    (X_train_orig_scaled, y_train_orig), 
    test_datasets_scaled, 
    title="Tuned ElasticNet - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/elasticnet_orig_split_tuned.svg", 
    fit=False
)

# ElasticNet - RandomizedSearchCV - CleanSplit
random_search_clean = RandomizedSearchCV(
    estimator=ElasticNet(random_state=42),
    param_distributions=param_dist_elasticnet,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_clean,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)

random_search_clean.fit(X_train_clean_scaled, y_train_clean)
save_cv_results(random_search_clean, f"elasticnet_clean_split_cv_results.csv")

print("ElasticNet Clean Split - Best params:", random_search_clean.best_params_)
print(f"ElasticNet Clean Split - Best CV R²: {random_search_clean.best_score_:.3f}")

best_elasticnet_clean = random_search_clean.best_estimator_

# Save predictions
save_predictions(best_elasticnet_clean, test_datasets_clean_scaled, pdb_ids_clean, "ElasticNet_tuned_clean")

evaluate_model(
    best_elasticnet_clean,
    (X_train_clean_scaled, y_train_clean), 
    test_datasets_clean_scaled, 
    title="Tuned ElasticNet - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path=f"{PLOTS_DIR}/elasticnet_clean_split_tuned.svg", 
    fit=False
)

# ==============================================================================
# 3. HIST GRADIENT BOOSTING (HGB)
# ==============================================================================

# HGB Baseline - Original Split
hgb_model = HistGradientBoostingRegressor(
    learning_rate=0.01,
    max_iter=300,
    max_depth=6,
    random_state=42
)

evaluate_model(
    hgb_model,
    (X_train_orig_scaled, y_train_orig),
    test_datasets_scaled,
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/hist_orig_split.svg"
)

# HGB - RandomizedSearchCV - Original Split
random_search_hgb_orig = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42),
    param_distributions=param_dist_hgb,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_orig,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)

random_search_hgb_orig.fit(X_train_orig_scaled, y_train_orig)
save_cv_results(random_search_hgb_orig, f"hgb_orig_split_cv_results.csv")

print("HGB Original Split - Best params:", random_search_hgb_orig.best_params_)
print(f"HGB Original Split - Best CV R²: {random_search_hgb_orig.best_score_:.3f}")

best_hgb_orig = random_search_hgb_orig.best_estimator_

# Save predictions
save_predictions(best_hgb_orig, test_datasets_scaled, pdb_ids_orig, "HGB_tuned_orig")

evaluate_model(
    best_hgb_orig,
    (X_train_orig_scaled, y_train_orig),
    test_datasets_scaled,
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/hist_orig_split_tuned.svg", 
    fit=False
)

# HGB Baseline - CleanSplit
evaluate_model(
    hgb_model,
    (X_train_clean_scaled, y_train_clean),
    test_datasets_clean_scaled,
    title="HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path=f"{PLOTS_DIR}/hist_clean_split.svg"
)

# HGB - RandomizedSearchCV - CleanSplit
random_search_hgb_clean = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42),
    param_distributions=param_dist_hgb,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_clean,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)

random_search_hgb_clean.fit(X_train_clean_scaled, y_train_clean)
save_cv_results(random_search_hgb_clean, f"hgb_clean_split_cv_results.csv")

print("HGB Clean Split - Best params:", random_search_hgb_clean.best_params_)
print(f"HGB Clean Split - Best CV R²: {random_search_hgb_clean.best_score_:.3f}")

best_hgb_clean = random_search_hgb_clean.best_estimator_

# Save predictions
save_predictions(best_hgb_clean, test_datasets_clean_scaled, pdb_ids_clean, "HGB_tuned_clean")

evaluate_model(
    best_hgb_clean,
    (X_train_clean_scaled, y_train_clean),
    test_datasets_clean_scaled,
    title="Tuned HistGradientBoostingRegressor - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path=f"{PLOTS_DIR}/hist_clean_split_tuned.svg", 
    fit=False
)

# ==============================================================================
# 4. XGBOOST
# ==============================================================================

# XGBoost Baseline - Original Split
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cuda", 
    random_state=42
)

evaluate_model(
    xgb_model,
    (X_train_orig_scaled, y_train_orig),
    test_datasets_scaled,
    title="XGBoost Regressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/xgb_orig_split.svg"
)

# XGBoost - RandomizedSearchCV - Original Split
random_search_xgb_orig = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_orig,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

random_search_xgb_orig.fit(X_train_orig_scaled, y_train_orig)
save_cv_results(random_search_xgb_orig, f"xgb_orig_split_cv_results.csv")

print("XGBoost Original Split - Best params:", random_search_xgb_orig.best_params_)
print(f"XGBoost Original Split - Best R²: {random_search_xgb_orig.best_score_:.4f}")

best_xgb_orig = random_search_xgb_orig.best_estimator_

# Save predictions
save_predictions(best_xgb_orig, test_datasets_scaled, pdb_ids_orig, "XGB_tuned_orig")

evaluate_model(
    best_xgb_orig,
    (X_train_orig_scaled, y_train_orig),
    test_datasets_scaled,
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Embeddings) (Original Split)",
    save_path=f"{PLOTS_DIR}/xgb_orig_split_tuned.svg", 
    fit=False
)

# XGBoost Baseline - CleanSplit
evaluate_model(
    xgb_model,
    (X_train_clean_scaled, y_train_clean),
    test_datasets_clean_scaled,
    title="XGBoost Regressor - Performance on CASF Test Sets (Clean Split)",
    save_path=f"{PLOTS_DIR}/xgb_clean_split.svg"
)

# XGBoost - RandomizedSearchCV - CleanSplit
random_search_xgb_clean = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=20,
    scoring='r2',
    cv=cv_splits_clean,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

random_search_xgb_clean.fit(X_train_clean_scaled, y_train_clean)
save_cv_results(random_search_xgb_clean, f"xgb_clean_split_cv_results.csv")

print("XGBoost Clean Split - Best params:", random_search_xgb_clean.best_params_)
print(f"XGBoost Clean Split - Best R²: {random_search_xgb_clean.best_score_:.4f}")

best_xgb_clean = random_search_xgb_clean.best_estimator_

# Save predictions
save_predictions(best_xgb_clean, test_datasets_clean_scaled, pdb_ids_clean, "XGB_tuned_clean")

evaluate_model(
    best_xgb_clean,
    (X_train_clean_scaled, y_train_clean),
    test_datasets_clean_scaled,
    title="Tuned XGBoost Regressor - Performance on CASF Test Sets (Embeddings) (Clean Split)",
    save_path=f"{PLOTS_DIR}/xgb_clean_split_tuned.svg", 
    fit=False
)

print("\n" + "="*80)
print("ALL EMBEDDINGS MODELS TRAINED AND PREDICTIONS SAVED!")
print("="*80)