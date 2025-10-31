import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def save_plot(fig, filename):
    """Save matplotlib figure as SVG in the PLOTS_DIR."""
    path = os.path.join(PLOTS_DIR, f"{filename}.svg") 
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")

def save_cv_results(random_search, split_name, model_name):
    """Save CV results (cv_results_) of a RandomizedSearchCV to CSV."""
    df = pd.DataFrame(random_search.cv_results_)
    path = os.path.join(CV_DIR, f"{model_name}_{split_name}_cv_results.csv")
    df.to_csv(path, index=False)
    print(f"CV results for {split_name.upper()} ({model_name}) saved to {path}")

def plot_results(test_datasets, results_dict, title, filename=None, figsize=(16, 10)):
    """Universal scatter plot for regression model results."""
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    for ax, (name, (X_test, y_test)) in zip(axes, test_datasets.items()):
        entry = results_dict[name]
        model = entry.get("model") or entry.get("best")
        y_pred = model.predict(X_test)

        r2, rmse, pcc = entry["R²"], entry["RMSE"], entry["PCC"]

        ax.scatter(y_test, y_pred, alpha=0.6)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--")
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 12)
        ax.set_title(f"{name.upper()}\nR²={r2:.3f}, RMSE={rmse:.3f}, PCC={pcc:.3f}")
        ax.set_xlabel("True pK")
        ax.set_ylabel("Predicted pK")
        ax.grid(True, linestyle="--", alpha=0.5)

    for ax in axes[len(test_datasets):]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=15, y=1.03)
    plt.tight_layout()
    if filename:
        save_plot(fig, filename)
    else:
        plt.show()

# -----------------------------------------------------------------------------
# Directory setup
# -----------------------------------------------------------------------------
BASE_DIR = "OOD_ML_results"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CV_DIR = os.path.join(BASE_DIR, "cv_results")

for d in [BASE_DIR, PLOTS_DIR, MODELS_DIR, CV_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Results will be stored in: {os.path.abspath(BASE_DIR)}")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
drop_cols = ["protein_sequences", "pocket_sequences", "ligand_smiles"]

df_all_encoded = pd.read_csv("PDBbind_protein_pocket_ligands_bindingsites_encoded.csv")
df_all_encoded = df_all_encoded.drop(columns=drop_cols).dropna().reset_index(drop=True)

# Load OOD splits
ood_splits_path = "PDBbind_ood_splits"
split_files = [f for f in os.listdir(ood_splits_path) if f.endswith(".json")]

train_datasets, test_datasets, train_df_dict = {}, {}, {}

for file in split_files:
    split_name = file.split("_")[1]
    with open(os.path.join(ood_splits_path, file), "r") as f:
        split = json.load(f)

    train_df = df_all_encoded[df_all_encoded["pdb_id"].isin(split["train"])]
    test_ids = {pdb for k, v in split.items() if k != "train" for pdb in v}
    test_df = df_all_encoded[df_all_encoded["pdb_id"].isin(test_ids)]

    train_df_dict[split_name] = train_df.copy()

    X_train, y_train = train_df.drop(columns=["pK", "pdb_id"]), train_df["pK"]
    X_test, y_test = test_df.drop(columns=["pK", "pdb_id"]), test_df["pK"]

    train_datasets[split_name] = (X_train, y_train)
    test_datasets[split_name] = (X_test, y_test)

print(f"Loaded {len(train_datasets)} OOD splits:")
for name, (Xtr, _) in train_datasets.items():
    print(f"   - {name}: {len(Xtr)} train / {len(test_datasets[name][0])} test samples")

# -----------------------------------------------------------------------------
# Validation Splits → CV Folds
# -----------------------------------------------------------------------------
val_splits_path = "val_splits_ood"
val_split_files = [f for f in os.listdir(val_splits_path) if f.endswith(".json")]

val_splits_dict = {}

for file in val_split_files:
    base = os.path.splitext(file)[0]
    parts = base.split("_")
    split_name, fold_name = parts[1], parts[-1]

    with open(os.path.join(val_splits_path, file), "r") as f:
        fold_data = json.load(f)

    val_splits_dict.setdefault(split_name, []).append({
        "fold": fold_name,
        "train": fold_data["train"],
        "validation": fold_data["validation"]
    })

def build_cv_splits(train_df, all_val_ids_list, id_col="pdb_id"):
    """Convert validation IDs into sklearn-compatible CV index tuples."""
    cv_splits = []
    id_to_idx = {pid: idx for idx, pid in enumerate(train_df[id_col])}
    for val_ids in all_val_ids_list:
        val_idx = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
        train_idx = [idx for pid, idx in id_to_idx.items() if pid not in val_ids]
        cv_splits.append((train_idx, val_idx))
    return cv_splits

cv_folds = {}
for split_name, train_df in train_df_dict.items():
    if split_name in val_splits_dict:
        val_ids_list = [fold["validation"] for fold in val_splits_dict[split_name]]
        cv_folds[split_name] = build_cv_splits(train_df, val_ids_list)
    else:
        print(f"No validation splits found for {split_name}")

# -----------------------------------------------------------------------------
# Baseline ElasticNet
# -----------------------------------------------------------------------------
results_lr, models_lr = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    X_test, y_test = test_datasets[name]
    model = ElasticNet(random_state=42, max_iter=10000).fit(X_train, y_train) 
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_lr[name] = {"model": model, "R²": r2, "RMSE": rmse, "PCC": pcc}
    models_lr[name] = model

plot_results(test_datasets, results_lr,
             "ElasticNet per OOD Split",
             filename="elasticnet_ood")

# -----------------------------------------------------------------------------
# ElasticNet — RandomizedSearchCV 
# -----------------------------------------------------------------------------
param_dist_elasticnet = {
    'alpha': uniform(0.001, 100.0),
    'l1_ratio': uniform(0.0, 1.0)
}

results_elasticnet_tuned, best_models_elasticnet = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    print(f"\nTuning ElasticNet for {name.upper()}...")

    random_search = RandomizedSearchCV(
        estimator=ElasticNet(random_state=42,max_iter=10000), 
        param_distributions=param_dist_elasticnet,
        n_iter=20,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
        random_state=42,
        cv=cv_folds[name]
    )
    random_search.fit(X_train, y_train)
    # Annahme: save_cv_results nutzt random_search.cv_results_
    save_cv_results(random_search, name, "ElasticNet") 

    # 1. Das beste Modell ist bereits gefittet und verfügbar:
    best_model = random_search.best_estimator_ 
    best_params = random_search.best_params_
    
    # 2. Direkte Vorhersage ohne erneutes Fitten:
    X_test, y_test = test_datasets[name]
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_elasticnet_tuned[name] = {
        "best": best_model,
        "R²": r2,
        "RMSE": rmse,
        "PCC": pcc,
        "CV_R²": random_search.best_score_,
        "BestParams": best_params
    }

plot_results(test_datasets, results_elasticnet_tuned,
             "ElasticNet — Tuned per OOD Split",
             filename="elasticnet_tuned_ood")
# -----------------------------------------------------------------------------
# HistGradientBoosting — Baseline
# -----------------------------------------------------------------------------
results_hgb, models_hgb = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    X_test, y_test = test_datasets[name]
    model = HistGradientBoostingRegressor(
        learning_rate=0.01,
        max_iter=300,
        max_depth=6,
        early_stopping=True,
        random_state=42
    ).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_hgb[name] = {"model": model, "R²": r2, "RMSE": rmse, "PCC": pcc}
    models_hgb[name] = model

plot_results(test_datasets, results_hgb,
             "HistGradientBoosting Regression per OOD Split",
             filename="hgb_ood")

# -----------------------------------------------------------------------------
# HistGradientBoosting — RandomizedSearchCV
# -----------------------------------------------------------------------------
param_dist_hgb = {
    "learning_rate": uniform(0.005, 0.05),
    "max_iter": randint(100, 600),
    "max_depth": randint(3, 10),
    "min_samples_leaf": randint(10, 80),
    "l2_regularization": uniform(0.0, 1.0),
    "max_bins": randint(100, 255)
}

results_hgb_tuned, best_models_hgb = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    print(f"\nTuning HGB for {name.upper()}...")

    random_search = RandomizedSearchCV(
        estimator=HistGradientBoostingRegressor(random_state=42),
        param_distributions=param_dist_hgb,
        n_iter=20,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
        random_state=42,
        cv=cv_folds[name]
    )
    random_search.fit(X_train, y_train)
    save_cv_results(random_search, name, "HistGB")

    best_params = random_search.best_params_
    best_model = HistGradientBoostingRegressor(**best_params, random_state=42, early_stopping=True)
    best_model.fit(X_train, y_train)

    X_test, y_test = test_datasets[name]
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_hgb_tuned[name] = {
        "best": best_model,
        "R²": r2,
        "RMSE": rmse,
        "PCC": pcc,
        "CV_R²": random_search.best_score_,
        "BestParams": best_params
    }

plot_results(test_datasets, results_hgb_tuned,
             "HistGradientBoosting — Tuned per OOD Split",
             filename="hgb_tuned_ood")
# -----------------------------------------------------------------------------
# XGBoost — Baseline
# -----------------------------------------------------------------------------
results_xgb_baseline, models_xgb_baseline = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    X_test, y_test = test_datasets[name]
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        random_state=42
    ).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_xgb_baseline[name] = {"model": model, "R²": r2, "RMSE": rmse, "PCC": pcc}
    models_xgb_baseline[name] = model

plot_results(test_datasets, results_xgb_baseline,
             "XGBoost Baseline per OOD Split",
             filename="xgb_baseline_ood")

# -----------------------------------------------------------------------------
# XGBoost — RandomizedSearchCV
# -----------------------------------------------------------------------------
base_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42
}

param_dist_xgb = {
    "n_estimators": randint(300, 1200),
    "learning_rate": loguniform(1e-3, 0.1),
    "max_depth": randint(3, 8),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "reg_alpha": loguniform(1e-3, 10),
    "reg_lambda": loguniform(0.1, 50)
}

results_xgb_tuned = {}

for name, (X_train, y_train) in train_datasets.items():
    print(f"\nTuning XGBoost for {name.upper()}...")
    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(**base_params),
        param_distributions=param_dist_xgb,
        n_iter=20,
        scoring="r2",
        cv=cv_folds[name],
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    save_cv_results(random_search, name, "XGBoost")

    best_params = random_search.best_params_
    best_model = XGBRegressor(**base_params, **best_params)
    best_model.fit(X_train, y_train)

    X_test, y_test = test_datasets[name]
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_xgb_tuned[name] = {
        "best": best_model,
        "R²": r2,
        "RMSE": rmse,
        "PCC": pcc,
        "CV_R²": random_search.best_score_,
        "BestParams": best_params
    }

plot_results(test_datasets, results_xgb_tuned,
             "XGBoost — Tuned per OOD Split",
             filename="xgb_tuned_ood")
