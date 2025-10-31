import os
import json
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
gc.collect()
os.makedirs("plots_molformer", exist_ok=True)

# --------------------------------------------------------------------------
# Helper: Save plots cleanly into plots_molformer/
# --------------------------------------------------------------------------
def save_plot(filename, title=None):
    """
    Saves the current matplotlib figure as SVG in 'plots_molformer/'.
    """
    if title:
        plt.suptitle(title, fontsize=15, y=1.03)
    plt.tight_layout()
    path = os.path.join("plots_molformer", f"{filename}.svg")
    plt.savefig(path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
df = pd.read_pickle("PDBbind_protein_ligands_embeddings_min_MoLFormer.pkl")

ood_splits_path = "PDBbind_ood_splits"
split_files = [f for f in os.listdir(ood_splits_path) if f.endswith(".json")]

train_datasets, test_datasets, train_df_dict = {}, {}, {}

for file in split_files:
    split_name = file.split("_")[1]
    with open(os.path.join(ood_splits_path, file), "r") as f:
        split = json.load(f)

    train_df = df[df["pdb_id"].isin(split["train"])]
    test_ids = {pdb for k, v in split.items() if k != "train" for pdb in v}
    test_df = df[df["pdb_id"].isin(test_ids)]

    train_df_dict[split_name] = train_df.copy()

    X_train = train_df.drop(columns=["pK", "pdb_id"])
    y_train = train_df["pK"]
    X_test = test_df.drop(columns=["pK", "pdb_id"])
    y_test = test_df["pK"]

    train_datasets[split_name] = (X_train, y_train)
    test_datasets[split_name] = (X_test, y_test)

print(f"Loaded {len(train_datasets)} OOD splits:")
for name, (Xtr, _) in train_datasets.items():
    print(f" - {name}: {len(Xtr)} train / {len(test_datasets[name][0])} test samples")

# --------------------------------------------------------------------------
# Concatenate embeddings
# --------------------------------------------------------------------------
for split_name, (X_train, y_train) in train_datasets.items():
    X_train = np.vstack([
        np.hstack([row["protein_embedding"], row["molformer_embedding"]])
        for _, row in X_train.iterrows()
    ])
    X_test = np.vstack([
        np.hstack([row["protein_embedding"], row["molformer_embedding"]])
        for _, row in test_datasets[split_name][0].iterrows()
    ])
    y_train = y_train.values
    y_test = test_datasets[split_name][1].values
    train_datasets[split_name] = (X_train, y_train)
    test_datasets[split_name] = (X_test, y_test)

# --------------------------------------------------------------------------
# Load Validation splits
# --------------------------------------------------------------------------
val_splits_path = "val_splits_ood"
val_split_files = [f for f in os.listdir(val_splits_path) if f.endswith(".json")]

val_splits_dict = {}
for file in val_split_files:
    base = os.path.splitext(file)[0]
    parts = base.split("_")
    split_name = parts[1]
    fold_name = parts[-1]
    with open(os.path.join(val_splits_path, file), "r") as f:
        fold_data = json.load(f)
    if split_name not in val_splits_dict:
        val_splits_dict[split_name] = []
    val_splits_dict[split_name].append({
        "fold": fold_name,
        "train": fold_data["train"],
        "validation": fold_data["validation"]
    })

# Helper: build sklearn CV folds
def build_cv_splits(train_df, all_val_ids_list, id_col="pdb_id"):
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
        print(f"Warning: No validation splits found for {split_name}")

# --------------------------------------------------------------------------
# Baseline ElasticNet
# --------------------------------------------------------------------------
results_lr, models_lr = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    X_test, y_test = test_datasets[name]
    model = ElasticNet(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]
    results_lr[name] = {"R²": r2, "RMSE": rmse, "PCC": pcc}
    models_lr[name] = (model, y_test, y_pred)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()
for ax, (name, (model, y_test, y_pred)) in zip(axes, models_lr.items()):
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    res = results_lr[name]
    ax.set_title(f"{name.upper()}\nR²={res['R²']:.3f}, RMSE={res['RMSE']:.3f}, PCC={res['PCC']:.3f}")
    ax.set_xlabel("True pK")
    ax.set_ylabel("Predicted pK")
    ax.grid(True, linestyle="--", alpha=0.5)

for ax in axes[len(test_datasets):]:
    ax.set_visible(False)

save_plot("elasticnet_baseline_per_split", "ElasticNet per OOD Split")
# --------------------------------------------------------------------------
# ElasticNet tuned
# --------------------------------------------------------------------------
param_dist = {
    'alpha': uniform(0.001, 100.0), 
    'l1_ratio': uniform(0.0, 1.0)    
}

best_models, results_elasticnet_tuned = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    print(f"\nTraining {name.upper()} split...")
    base_model = ElasticNet(random_state=42, max_iter=10000) 
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
        random_state=42,
        cv=cv_folds[name]
    )
    random_search.fit(X_train, y_train)
    # Save CV results
    cv_elasticnet_df = pd.DataFrame(random_search.cv_results_)
    cv_elasticnet_df.to_csv(f"plots_molformer/elasticnet_cv_results_{name}.csv", index=False)
    
    best_model = random_search.best_estimator_ 
    best_params = random_search.best_params_

    best_models[name] = best_model
    X_test, y_test = test_datasets[name]
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]
    
    results_elasticnet_tuned[name] = {
        "BestParams": best_params,
        "CV_R²": random_search.best_score_,
        "Test_R²": r2,
        "Test_RMSE": rmse,
        "Test_PCC": pcc
    }
    print(f"{name.upper()} — CV R²={random_search.best_score_:.3f}, Test R²={r2:.3f}\n")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, best_models.items()):
    X_test, y_test = test_datasets[name]
    y_pred = model.predict(X_test)
    res = results_elasticnet_tuned[name]
    
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    
    ax.set_title(f"{name.upper()}\nR²={res['Test_R²']:.3f}, RMSE={res['Test_RMSE']:.3f}, PCC={res['Test_PCC']:.3f}")
    ax.set_xlabel("True pK")
    ax.set_ylabel("Predicted pK")
    ax.grid(True, linestyle="--", alpha=0.5)

for ax in axes[len(test_datasets):]:
    ax.set_visible(False)

save_plot("elasticnet_tuned_per_split", "ElasticNet RandomizedSearchCV — Tuned per OOD Split")
#
# --------------------------------------------------------------------------
# HistGradientBoosting baseline
# --------------------------------------------------------------------------
results_hgb = {}
models_hgb = {}

for name, (X_train, y_train) in train_datasets.items():
    X_test, y_test = test_datasets[name]
    model = HistGradientBoostingRegressor(
        learning_rate=0.01,
        max_iter=300,
        max_depth=6,
        early_stopping=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]
    results_hgb[name] = {"R²": r2, "RMSE": rmse, "PCC": pcc}
    models_hgb[name] = (model, y_test, y_pred)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()
for ax, (name, (model, y_test, y_pred)) in zip(axes, models_hgb.items()):
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    res = results_hgb[name]
    ax.set_title(f"{name.upper()}\nR²={res['R²']:.3f}, RMSE={res['RMSE']:.3f}, PCC={res['PCC']:.3f}")
    ax.set_xlabel("True pK")
    ax.set_ylabel("Predicted pK")
    ax.grid(True, linestyle="--", alpha=0.5)

for ax in axes[len(test_datasets):]:
    ax.set_visible(False)

save_plot("hgb_baseline_per_split", "HistGradientBoosting Regression per OOD Split")

# --------------------------------------------------------------------------
# HistGradientBoosting tuned
# --------------------------------------------------------------------------
param_dist = {
    "learning_rate": uniform(0.005, 0.05),
    "max_iter": randint(100, 600),
    "max_depth": randint(3, 10),
    "min_samples_leaf": randint(10, 80),
    "l2_regularization": uniform(0.0, 1.0),
    "max_bins": randint(100, 255)
}

best_models, results_hgb_tuned = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    print(f"\nTraining {name.upper()} split...")
    
    base_model = HistGradientBoostingRegressor(random_state=42) 
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
        random_state=42,
        cv=cv_folds[name]
    )
    random_search.fit(X_train, y_train)
    # Save CV results
    cv_hgb_df = pd.DataFrame(random_search.cv_results_)
    cv_hgb_df.to_csv(f"plots_molformer/hgb_cv_results_{name}.csv", index=False)
    
    best_model = random_search.best_estimator_ 
    best_params = random_search.best_params_
    
    best_models[name] = best_model
    X_test, y_test = test_datasets[name]
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]
    
    results_hgb_tuned[name] = {
        "BestParams": best_params,
        "CV_R²": random_search.best_score_,
        "Test_R²": r2,
        "Test_RMSE": rmse,
        "Test_PCC": pcc
    }
    print(f"{name.upper()} — CV R²={random_search.best_score_:.3f}, Test R²={r2:.3f}\n")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()
for ax, (name, model) in zip(axes, best_models.items()):
    X_test, y_test = test_datasets[name]
    y_pred = model.predict(X_test)
    res = results_hgb_tuned[name]
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    ax.set_title(f"{name.upper()}\nR²={res['Test_R²']:.3f}, RMSE={res['Test_RMSE']:.3f}, PCC={res['Test_PCC']:.3f}")
    ax.set_xlabel("True pK")
    ax.set_ylabel("Predicted pK")
    ax.grid(True, linestyle="--", alpha=0.5)

for ax in axes[len(test_datasets):]:
    ax.set_visible(False)

save_plot("hgb_tuned_per_split", "HistGradientBoosting RandomizedSearchCV — Tuned per OOD Split")

# --------------------------------------------------------------------------
# XGBoost baseline
# --------------------------------------------------------------------------
results_xgb = {}
models_xgb = {}

for name, (X_train, y_train) in train_datasets.items():
    X_test, y_test = test_datasets[name]

    model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        device='cuda',
        random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_xgb[name] = {"R²": r2, "RMSE": rmse, "PCC": pcc}
    models_xgb[name] = (model, y_test, y_pred)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()
for ax, (name, (model, y_test, y_pred)) in zip(axes, models_xgb.items()):
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    res = results_xgb[name]
    ax.set_title(f"{name.upper()}\nR²={res['R²']:.3f}, RMSE={res['RMSE']:.3f}, PCC={res['PCC']:.3f}")
    ax.set_xlabel("True pK")
    ax.set_ylabel("Predicted pK")
    ax.grid(True, linestyle="--", alpha=0.5)


for ax in axes[len(test_datasets):]:
    ax.set_visible(False)

save_plot("xgb_baseline_per_split", "XGBoost Regression per OOD Split")

# --------------------------------------------------------------------------
# XGBoost tuned (RandomizedSearchCV)
# --------------------------------------------------------------------------
param_dist_xgb = {
    'n_estimators': randint(300, 1200),            
    'learning_rate': loguniform(1e-3, 0.1),         
    'max_depth': randint(3, 8),                     
    'subsample': uniform(0.7, 0.3),                 
    'colsample_bytree': uniform(0.7, 0.3),          
    'reg_alpha': loguniform(1e-3, 10),              
    'reg_lambda': loguniform(0.1, 50)               
}

best_models_xgb, results_xgb_tuned = {}, {}

for name, (X_train, y_train) in train_datasets.items():
    print(f"\nTuning XGBoost for {name.upper()} split...")

    base_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        device='cuda',
        random_state=42)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist_xgb,
        n_iter=20,
        scoring='r2',
        cv=cv_folds[name],
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    # Save CV results
    cv_xgb_df = pd.DataFrame(random_search.cv_results_)
    cv_xgb_df.to_csv(f"plots_molformer/xgb_cv_results_{name}.csv", index=False)

    best_model = random_search.best_estimator_ 
    best_params = random_search.best_params_
    
    best_models_xgb[name] = best_model

    X_test, y_test = test_datasets[name]
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc = np.corrcoef(y_test, y_pred)[0, 1]

    results_xgb_tuned[name] = {
        "BestParams": best_params,
        "CV_R²": random_search.best_score_,
        "Test_R²": r2,
        "Test_RMSE": rmse,
        "Test_PCC": pcc
    }

    print(f"{name.upper()} — CV R²={random_search.best_score_:.3f}, Test R²={r2:.3f}\n")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()
for ax, (name, model) in zip(axes, best_models_xgb.items()):
    X_test, y_test = test_datasets[name]
    y_pred = model.predict(X_test)
    res = results_xgb_tuned[name]
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    ax.set_title(f"{name.upper()}\nR²={res['Test_R²']:.3f}, RMSE={res['Test_RMSE']:.3f}, PCC={res['Test_PCC']:.3f}")
    ax.set_xlabel("True pK")
    ax.set_ylabel("Predicted pK")
    ax.grid(True, linestyle="--", alpha=0.5)


for ax in axes[len(test_datasets):]:
    ax.set_visible(False)

save_plot("xgb_tuned_per_split", "XGBoost RandomizedSearchCV — Tuned per OOD Split")

# --------------------------------------------------------------------------
# Save results summary
# --------------------------------------------------------------------------
summary = {
    "HGB_Baseline": results_hgb,
    "HGB_Tuned": results_hgb_tuned,
    "XGB_Baseline": results_xgb,
    "XGB_Tuned": results_xgb_tuned
}

with open("plots_molformer/results_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("All models trained and plots saved under 'plots_molformer'")

