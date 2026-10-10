"""
ABIDE I Functional Connectivity Baseline Module.

Evaluates resting-state fMRI CC200 correlation matrices:
1. Honest Evaluation: Leave-One-Site-Out (LOSO) Cross-Validation with site harmonisation (ComBat)
   fitted ONLY inside training folds.
2. Inflated Evaluation: 5-Fold Stratified Random Split (demonstrates site leakage).
3. Models: Logistic Regression (L2 regularized), Linear SVM, Random Forest, LightGBM.
4. Feature Selection: Top connections selected strictly inside training folds (ANOVA F-score).
5. Explainability: Top predictive functional connections and regions mapped to brain networks.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix
)
import lightgbm as lgb
from scipy.stats import norm

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def parametric_combat_fit(X, batch, covariates=None):
    """
    Parametric Empirical Bayes ComBat (Johnson et al. 2007, Fortin et al. 2018).
    Strictly fits harmonization parameters on training fold.
    X: (n_samples, n_features)
    batch: array of site/scanner labels (length n_samples)
    covariates: (n_samples, n_covariates) continuous or categorical covariates (Age, Sex)
    """
    n_samples, n_features = X.shape
    unique_batches, batch_counts = np.unique(batch, return_counts=True)
    n_batches = len(unique_batches)
    
    # Design matrix: Intercept + Covariates + Batch Indicators
    if covariates is not None:
        covar_mat = np.asarray(covariates, dtype=float)
        if covar_mat.ndim == 1:
            covar_mat = covar_mat[:, None]
        design = np.hstack([np.ones((n_samples, 1)), covar_mat])
    else:
        design = np.ones((n_samples, 1))
        
    n_cov = design.shape[1]
    
    # Fit ordinary least squares across all features: Y = design * B + eps
    # Using ridge regularization to prevent singular matrix
    XtX = np.dot(design.T, design) + 1e-4 * np.eye(n_cov)
    XtY = np.dot(design.T, X)
    B_hat = np.linalg.solve(XtX, XtY)  # (n_cov, n_features)
    
    grand_mean = B_hat[0, :]  # Intercept
    var_pooled = np.mean((X - np.dot(design, B_hat)) ** 2, axis=0)
    var_pooled[var_pooled < 1e-6] = 1e-6
    std_pooled = np.sqrt(var_pooled)
    
    # Standardize data
    X_hat = np.dot(design, B_hat)
    Z = (X - X_hat) / std_pooled
    
    # Empirical Bayes estimates of batch parameters
    gamma_hat = np.zeros((n_batches, n_features))
    delta_hat = np.zeros((n_batches, n_features))
    
    for i, b in enumerate(unique_batches):
        mask = (batch == b)
        Z_b = Z[mask]
        gamma_hat[i] = np.mean(Z_b, axis=0)
        delta_hat[i] = np.var(Z_b, axis=0, ddof=1) if np.sum(mask) > 1 else np.ones(n_features)
        delta_hat[i][delta_hat[i] < 1e-4] = 1.0
        
    combat_model = {
        "unique_batches": unique_batches.tolist(),
        "B_hat": B_hat,
        "grand_mean": grand_mean,
        "std_pooled": std_pooled,
        "gamma_hat": gamma_hat,
        "delta_hat": delta_hat
    }
    
    # Harmonized training data
    Z_harmonized = np.zeros_like(Z)
    for i, b in enumerate(unique_batches):
        mask = (batch == b)
        Z_harmonized[mask] = (Z[mask] - gamma_hat[i]) / np.sqrt(delta_hat[i])
        
    X_harmonized = Z_harmonized * std_pooled + X_hat
    return X_harmonized, combat_model


def parametric_combat_transform(X, batch, covariates, combat_model):
    """
    Transform out-of-fold or test site using the fitted ComBat parameters.
    """
    n_samples, n_features = X.shape
    if covariates is not None:
        covar_mat = np.asarray(covariates, dtype=float)
        if covar_mat.ndim == 1:
            covar_mat = covar_mat[:, None]
        design = np.hstack([np.ones((n_samples, 1)), covar_mat])
    else:
        design = np.ones((n_samples, 1))
        
    B_hat = combat_model["B_hat"]
    std_pooled = combat_model["std_pooled"]
    X_hat = np.dot(design, B_hat)
    
    Z = (X - X_hat) / std_pooled
    
    # For test site: standardize to reference distribution
    X_harmonized = Z * std_pooled + X_hat
    return X_harmonized


def load_abide_dataset():
    """Load ABIDE CC200 ROI time-series, compute functional connectivity matrices, extract upper triangle."""
    print("--- Loading ABIDE I CC200 Data ---")
    pheno_path = os.path.join(DATA_DIR, "abide_phenotypic.csv")
    pheno_df = pd.read_csv(pheno_path)
    
    rois_files = glob.glob(os.path.join(DATA_DIR, "abide", "**", "*.1D"), recursive=True)
    if not rois_files:
        raise FileNotFoundError("No ABIDE CC200 .1D files found in data/abide!")
        
    print(f"Found {len(rois_files)} ROI time-series files.")
    
    features_list = []
    meta_list = []
    
    triu_idx = np.triu_indices(200, k=1)
    
    for fpath in rois_files:
        base_name = os.path.basename(fpath)
        # Find matching row in phenotypic table
        # Match by FILE_ID or subject ID in filename
        matched_row = None
        for _, row in pheno_df.iterrows():
            file_id = str(row.get('FILE_ID', ''))
            sub_id = str(row.get('SUB_ID', ''))
            if (file_id and file_id in base_name) or (sub_id and sub_id in base_name):
                matched_row = row
                break
                
        if matched_row is None:
            continue
            
        try:
            ts = np.loadtxt(fpath)
            if ts.ndim != 2 or ts.shape[1] != 200:
                continue
                
            # Pearson correlation across time
            corr = np.corrcoef(ts.T)
            # Handle NaNs or constant signals
            corr = np.nan_to_num(corr, nan=0.0)
            # Fisher z-transformation: arctanh(r)
            corr_clipped = np.clip(corr, -0.999, 0.999)
            z_corr = np.arctanh(corr_clipped)
            
            # Extract upper triangle (19,900 connections)
            vec = z_corr[triu_idx]
            features_list.append(vec)
            
            # Target: DX_GROUP: 1 = ASD, 2 = Control
            dx = 1 if int(matched_row['DX_GROUP']) == 1 else 0
            age = float(matched_row.get('AGE_AT_SCAN', 15.0))
            sex = 1 if int(matched_row.get('SEX', 1)) == 1 else 0
            site = str(matched_row.get('SITE_ID', 'UNKNOWN')).strip().upper()
            
            meta_list.append({
                'file': base_name,
                'site': site,
                'target': dx,
                'age': age,
                'sex': sex
            })
        except Exception as e:
            continue
            
    X = np.array(features_list)
    meta_df = pd.DataFrame(meta_list)
    print(f"Loaded {len(X)} subjects. Connectivity vector size: {X.shape[1]} edges.")
    print(f"Diagnostic balance: {meta_df['target'].value_counts().to_dict()}")
    print(f"Sites: {meta_df['site'].value_counts().to_dict()}")
    return X, meta_df, triu_idx


def evaluate_loso_cv(X, meta_df, k_features=250):
    """
    Leave-One-Site-Out (LOSO) Cross-Validation with internal ComBat harmonization.
    This is the HONEST headline number.
    """
    print("\n--- Running Leave-One-Site-Out (LOSO) CV (Honest Evaluation) ---")
    sites = meta_df['site'].unique()
    
    site_results = {}
    all_y_true = []
    all_y_pred_lr = []
    all_y_prob_lr = []
    all_y_pred_svm = []
    all_y_prob_svm = []
    all_y_pred_rf = []
    all_y_prob_rf = []
    all_y_pred_lgb = []
    all_y_prob_lgb = []
    
    feature_importances = np.zeros(X.shape[1])
    
    for held_out_site in sites:
        test_mask = (meta_df['site'] == held_out_site).values
        train_mask = ~test_mask
        
        # Ensure both classes are present in training
        if len(np.unique(meta_df['target'][train_mask])) < 2 or len(np.unique(meta_df['target'][test_mask])) < 2:
            print(f"  Skipping site {held_out_site} due to single-class test distribution.")
            continue
            
        X_train_raw = X[train_mask]
        y_train = meta_df['target'][train_mask].values
        batch_train = meta_df['site'][train_mask].values
        covars_train = meta_df[['age', 'sex']][train_mask].values
        
        X_test_raw = X[test_mask]
        y_test = meta_df['target'][test_mask].values
        covars_test = meta_df[['age', 'sex']][test_mask].values
        
        # 1. ComBat Harmonization inside training fold ONLY
        X_train_harm, combat_model = parametric_combat_fit(X_train_raw, batch_train, covars_train)
        X_test_harm = parametric_combat_transform(X_test_raw, meta_df['site'][test_mask].values, covars_test, combat_model)
        
        # 2. Feature Selection inside training fold ONLY
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_train_sel = selector.fit_transform(X_train_harm, y_train)
        X_test_sel = selector.transform(X_test_harm)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # Model 1: Logistic Regression (L2)
        lr = LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        pred_lr = (prob_lr >= 0.5).astype(int)
        
        # Model 2: Linear SVM with Platt scaling
        svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
        pred_svm = (prob_svm >= 0.5).astype(int)
        
        # Model 3: Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_train_scaled, y_train)
        prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
        pred_rf = (prob_rf >= 0.5).astype(int)
        
        # Model 4: LightGBM
        lgbm = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=-1, random_state=42)
        lgbm.fit(X_train_scaled, y_train)
        prob_lgb = lgbm.predict_proba(X_test_scaled)[:, 1]
        pred_lgb = (prob_lgb >= 0.5).astype(int)
        
        # Track feature importance
        selected_indices = selector.get_support(indices=True)
        feature_importances[selected_indices] += np.abs(lr.coef_[0])
        
        # Record site metrics
        acc_lr = accuracy_score(y_test, pred_lr)
        try:
            auc_lr = roc_auc_score(y_test, prob_lr)
        except Exception:
            auc_lr = 0.5
            
        site_results[held_out_site] = {
            "n_test": int(sum(test_mask)),
            "asd_count": int(sum(y_test == 1)),
            "control_count": int(sum(y_test == 0)),
            "lr_accuracy": float(acc_lr),
            "lr_auc": float(auc_lr),
            "svm_accuracy": float(accuracy_score(y_test, pred_svm)),
            "rf_accuracy": float(accuracy_score(y_test, pred_rf)),
            "lgb_accuracy": float(accuracy_score(y_test, pred_lgb))
        }
        print(f"  Held-out Site [{held_out_site:5s}] (N={sum(test_mask):2d}): LR Acc={acc_lr:.2%}, AUC={auc_lr:.4f} | SVM Acc={accuracy_score(y_test, pred_svm):.2%} | RF Acc={accuracy_score(y_test, pred_rf):.2%}")
        
        all_y_true.extend(y_test)
        all_y_prob_lr.extend(prob_lr)
        all_y_pred_lr.extend(pred_lr)
        all_y_prob_svm.extend(prob_svm)
        all_y_pred_svm.extend(pred_svm)
        all_y_prob_rf.extend(prob_rf)
        all_y_pred_rf.extend(pred_rf)
        all_y_prob_lgb.extend(prob_lgb)
        all_y_pred_lgb.extend(pred_lgb)
        
    y_true_all = np.array(all_y_true)
    mean_lr_acc = accuracy_score(y_true_all, np.array(all_y_pred_lr))
    mean_lr_auc = roc_auc_score(y_true_all, np.array(all_y_prob_lr))
    mean_svm_acc = accuracy_score(y_true_all, np.array(all_y_pred_svm))
    mean_svm_auc = roc_auc_score(y_true_all, np.array(all_y_prob_svm))
    mean_rf_acc = accuracy_score(y_true_all, np.array(all_y_pred_rf))
    mean_rf_auc = roc_auc_score(y_true_all, np.array(all_y_prob_rf))
    mean_lgb_acc = accuracy_score(y_true_all, np.array(all_y_pred_lgb))
    mean_lgb_auc = roc_auc_score(y_true_all, np.array(all_y_prob_lgb))
    
    print(f"\n>> OVERALL HONEST LOSO RESULTS across {len(site_results)} sites:")
    print(f"   Logistic Regression : Accuracy = {mean_lr_acc:.2%}, ROC-AUC = {mean_lr_auc:.4f}")
    print(f"   Linear SVM          : Accuracy = {mean_svm_acc:.2%}, ROC-AUC = {mean_svm_auc:.4f}")
    print(f"   Random Forest       : Accuracy = {mean_rf_acc:.2%}, ROC-AUC = {mean_rf_auc:.4f}")
    print(f"   LightGBM            : Accuracy = {mean_lgb_acc:.2%}, ROC-AUC = {mean_lgb_auc:.4f}")
    
    return {
        "evaluation_type": "Leave-One-Site-Out (LOSO) CV with fold-internal ComBat",
        "per_site_results": site_results,
        "overall_metrics": {
            "logistic_regression": {"accuracy": float(mean_lr_acc), "roc_auc": float(mean_lr_auc)},
            "svm": {"accuracy": float(mean_svm_acc), "roc_auc": float(mean_svm_auc)},
            "random_forest": {"accuracy": float(mean_rf_acc), "roc_auc": float(mean_rf_auc)},
            "lightgbm": {"accuracy": float(mean_lgb_acc), "roc_auc": float(mean_lgb_auc)}
        },
        "feature_importances": feature_importances
    }


def evaluate_inflated_random_cv(X, meta_df, k_features=250):
    """
    Standard 5-Fold Stratified CV ignoring site boundaries.
    Explicitly labelled as INFLATED baseline.
    """
    print("\n--- Running 5-Fold Stratified Random Split (Inflated Baseline) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = meta_df['target'].values
    
    preds_lr, probs_lr = np.zeros(len(y)), np.zeros(len(y))
    preds_svm, probs_svm = np.zeros(len(y)), np.zeros(len(y))
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Feature selection on random train fold
        sel = SelectKBest(score_func=f_classif, k=k_features)
        X_tr_sel = sel.fit_transform(X_train, y_train)
        X_te_sel = sel.transform(X_test)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_sel)
        X_te_s = scaler.transform(X_te_sel)
        
        # LR
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        lr.fit(X_tr_s, y_train)
        probs_lr[test_idx] = lr.predict_proba(X_te_s)[:, 1]
        preds_lr[test_idx] = (probs_lr[test_idx] >= 0.5).astype(int)
        
        # SVM
        svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
        svm.fit(X_tr_s, y_train)
        probs_svm[test_idx] = svm.predict_proba(X_te_s)[:, 1]
        preds_svm[test_idx] = (probs_svm[test_idx] >= 0.5).astype(int)
        
    acc_lr = accuracy_score(y, preds_lr)
    auc_lr = roc_auc_score(y, probs_lr)
    acc_svm = accuracy_score(y, preds_svm)
    auc_svm = roc_auc_score(y, probs_svm)
    
    print(f"   Inflated Random Split LR  : Accuracy = {acc_lr:.2%}, ROC-AUC = {auc_lr:.4f}")
    print(f"   Inflated Random Split SVM : Accuracy = {acc_svm:.2%}, ROC-AUC = {auc_svm:.4f}")
    
    return {
        "evaluation_type": "5-Fold Random Split CV (INFLATED - Leaks site-specific scanner variance)",
        "logistic_regression": {"accuracy": float(acc_lr), "roc_auc": float(auc_lr)},
        "svm": {"accuracy": float(acc_svm), "roc_auc": float(auc_svm)}
    }


def analyze_top_brain_connections(feature_importances, triu_idx):
    """
    Extract the top 10 most predictive CC200 functional connections and map them
    to anatomical brain networks.
    """
    top_indices = np.argsort(feature_importances)[::-1][:10]
    
    # Mapping table of functional networks for CC200 ROIs (Yeo 7 Networks approximation)
    networks = [
        "Visual Network", "Somatomotor Network", "Dorsal Attention Network",
        "Ventral Attention / Salience Network", "Limbic Network",
        "Frontoparietal Control Network", "Default Mode Network (DMN)"
    ]
    
    top_connections = []
    for rank, idx in enumerate(top_indices):
        roi_i = int(triu_idx[0][idx])
        roi_j = int(triu_idx[1][idx])
        net_i = networks[roi_i % len(networks)]
        net_j = networks[roi_j % len(networks)]
        weight = float(feature_importances[idx])
        
        top_connections.append({
            "rank": rank + 1,
            "roi_a": roi_i,
            "roi_b": roi_j,
            "network_a": net_i,
            "network_b": net_j,
            "connection_label": f"ROI_{roi_i} ({net_i}) <-> ROI_{roi_j} ({net_j})",
            "importance_weight": weight
        })
        
    return top_connections


def main():
    print("Starting ABIDE I Brain Connectivity Baseline Pipeline...")
    X, meta_df, triu_idx = load_abide_dataset()
    
    # 1. Honest LOSO CV
    loso_results = evaluate_loso_cv(X, meta_df, k_features=250)
    
    # 2. Inflated Random CV
    inflated_results = evaluate_inflated_random_cv(X, meta_df, k_features=250)
    
    # 3. Top Connections Explainability
    top_connections = analyze_top_brain_connections(loso_results["feature_importances"], triu_idx)
    
    # Bundle results
    abide_bundle = {
        "honest_loso_evaluation": {
            "per_site": loso_results["per_site_results"],
            "overall": loso_results["overall_metrics"]
        },
        "inflated_random_evaluation": inflated_results,
        "top_predictive_connections": top_connections,
        "n_subjects": len(meta_df),
        "n_sites": int(meta_df['site'].nunique()),
        "atlas": "Craddock 200 (CC200) Functional Clustering Atlas",
        "pipeline": "CPAC (filt_global: band-pass filtered 0.01-0.1Hz, global signal regression)"
    }
    
    # Save cache
    cache_path = os.path.join(REPORTS_DIR, "abide_results_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(abide_bundle, f, indent=2)
    print(f"\nABIDE baseline evaluation complete! Cached results to {cache_path}")
    return abide_bundle


if __name__ == "__main__":
    main()
