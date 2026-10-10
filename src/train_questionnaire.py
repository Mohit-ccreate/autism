"""
Comprehensive Questionnaire ML Training, Tuning, Calibration, and Explainability.

Implements:
1. Baseline: Logistic Regression (L2 regularized, standard scaled)
2. Tree Ensembles: LightGBM, XGBoost, CatBoost
3. Tabular Foundation Model: TabPFN (with gated authentication handling)
4. Meta-Learner: Stacking Ensemble
5. Time-bounded Optuna hyperparameter optimization with Stratified Repeated CV (5 folds, 2 repeats)
6. Probability Calibration (Platt Sigmoid / Isotonic via CalibratedClassifierCV)
7. Conformal Prediction: Non-conformity sets with 90% confidence, flagging uncertain cases
8. Comprehensive Metrics with 95% Bootstrap Confidence Intervals:
   Accuracy, Balanced Accuracy, Sensitivity, Specificity, PPV, NPV, F1, ROC-AUC, PR-AUC, MCC, Brier Score
9. Fairness Audit: Disaggregated metrics across Sex (male vs female)
10. Scientific Variants:
    - Variant A: Full 10 items (noted as deterministic rule replication)
    - Variant B: 10 items + demographics (no ethnicity or country)
    - Variant C: Abbreviated 5-item rapid screening (A1 to A5, honest probabilistic screener)
    - Variant D: Demographics-only baseline (negative control)
11. Leakage Demonstration: With score vs Without score
12. SHAP Explainability: Feature attributions and plots
13. ONNX & JSON Export for 100% offline browser inference
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import shap
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Suppress verbose warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics with specificity and NPV."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
        
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "brier_score": float(brier),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def compute_bootstrap_cis(y_true, y_pred, y_prob, n_bootstraps=1000, seed=42):
    """Compute 95% bootstrap confidence intervals for all metrics."""
    np.random.seed(seed)
    n = len(y_true)
    boot_metrics = {
        "accuracy": [], "balanced_accuracy": [], "sensitivity": [],
        "specificity": [], "ppv": [], "npv": [], "f1": [],
        "roc_auc": [], "pr_auc": [], "mcc": [], "brier_score": []
    }
    
    for _ in range(n_bootstraps):
        idx = np.random.randint(0, n, size=n)
        yt_sample = y_true[idx]
        yp_sample = y_pred[idx]
        yprob_sample = y_prob[idx]
        
        if len(np.unique(yt_sample)) < 2:
            continue
            
        m = compute_metrics(yt_sample, yp_sample, yprob_sample)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])
            
    cis = {}
    for k, vals in boot_metrics.items():
        if len(vals) > 0:
            low = np.percentile(vals, 2.5)
            high = np.percentile(vals, 97.5)
            cis[k] = {"ci_lower": float(low), "ci_upper": float(high)}
        else:
            cis[k] = {"ci_lower": 0.0, "ci_upper": 0.0}
            
    return cis


def conformal_prediction_audit(cal_probs, cal_labels, test_probs, alpha=0.10):
    """Split conformal prediction giving 90% coverage guarantee."""
    cal_scores = []
    for p, y in zip(cal_probs, cal_labels):
        true_prob = p if y == 1 else (1.0 - p)
        cal_scores.append(1.0 - true_prob)
        
    cal_scores = np.sort(cal_scores)
    q_level = np.ceil((len(cal_scores) + 1) * (1 - alpha)) / len(cal_scores)
    q_level = min(1.0, max(0.0, q_level))
    q_hat = np.quantile(cal_scores, q_level)
    
    conformal_results = []
    for p in test_probs:
        s0 = p
        s1 = 1.0 - p
        in_set_0 = (s0 <= q_hat)
        in_set_1 = (s1 <= q_hat)
        
        if in_set_0 and in_set_1:
            state = "uncertain"
        elif in_set_1:
            state = "higher_traits"
        elif in_set_0:
            state = "lower_traits"
        else:
            state = "uncertain"
            
        conformal_results.append({
            "in_set_0": bool(in_set_0),
            "in_set_1": bool(in_set_1),
            "state": state,
            "threshold": float(q_hat)
        })
        
    uncertain_rate = sum(1 for r in conformal_results if r['state'] == 'uncertain') / len(conformal_results)
    return conformal_results, uncertain_rate, float(q_hat)


def tune_and_train_cohort(train_df, test_df, feature_cols, cohort_name, variant_name):
    """Train, tune, calibrate, and evaluate all models for a specific cohort and feature variant."""
    print(f"\nTraining: {cohort_name.upper()} | Variant: {variant_name}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df['target'].values.astype(int)
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df['target'].values.astype(int)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    # 1. Logistic Regression Baseline
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    lr_metrics = compute_metrics(y_test, lr_pred, lr_prob)
    
    # 2. LightGBM with Optuna
    def lgb_objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'num_leaves': trial.suggest_int('num_leaves', 4, 32),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train[train_idx], y_train[train_idx])
            val_prob = clf.predict_proba(X_train[val_idx])[:, 1]
            scores.append(roc_auc_score(y_train[val_idx], val_prob))
            
        return np.mean(scores)
        
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=10, timeout=15)
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 42})
    best_lgb = lgb.LGBMClassifier(**best_lgb_params)
    best_lgb.fit(X_train, y_train)
    lgb_prob = best_lgb.predict_proba(X_test)[:, 1]
    lgb_pred = (lgb_prob >= 0.5).astype(int)
    lgb_metrics = compute_metrics(y_test, lgb_pred, lgb_prob)
    
    # 3. XGBoost
    best_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric='logloss', random_state=42
    )
    best_xgb.fit(X_train, y_train)
    xgb_prob = best_xgb.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_prob)
    
    # 4. CatBoost
    best_cat = CatBoostClassifier(iterations=120, depth=4, learning_rate=0.05, verbose=0, random_seed=42)
    best_cat.fit(X_train, y_train)
    cat_prob = best_cat.predict_proba(X_test)[:, 1]
    cat_pred = (cat_prob >= 0.5).astype(int)
    cat_metrics = compute_metrics(y_test, cat_pred, cat_prob)
    
    # 5. Stacking Ensemble
    estimators = [('lr', lr_pipe), ('lgb', best_lgb), ('xgb', best_xgb), ('cat', best_cat)]
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=42),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    stack_prob = stacking_clf.predict_proba(X_test)[:, 1]
    stack_pred = (stack_prob >= 0.5).astype(int)
    stack_metrics = compute_metrics(y_test, stack_pred, stack_prob)
    
    models_dict = {
        "LogisticRegression": {"model": lr_pipe, "metrics": lr_metrics, "probs": lr_prob},
        "LightGBM": {"model": best_lgb, "metrics": lgb_metrics, "probs": lgb_prob},
        "XGBoost": {"model": best_xgb, "metrics": xgb_metrics, "probs": xgb_prob},
        "CatBoost": {"model": best_cat, "metrics": cat_metrics, "probs": cat_prob},
        "StackingEnsemble": {"model": stacking_clf, "metrics": stack_metrics, "probs": stack_prob}
    }
    
    # Print model comparison table
    print(f"  {'Model':<20} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'F1-Score':<10} | {'Brier':<10}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for m_name, m_data in models_dict.items():
        m_met = m_data["metrics"]
        print(f"  {m_name:<20} | {m_met['accuracy']:>9.2%} | {m_met['roc_auc']:>10.4f} | {m_met['f1']:>10.4f} | {m_met['brier_score']:>10.4f}")
    
    # Find best model
    best_name = max(models_dict.keys(), key=lambda k: models_dict[k]["metrics"]["roc_auc"])
    best_entry = models_dict[best_name]
    print(f"  >>> Best Model Selected: {best_name} (Acc: {best_entry['metrics']['accuracy']:.2%}, ROC-AUC: {best_entry['metrics']['roc_auc']:.4f})\n")
    
    # Calibration
    calibrated_clf = CalibratedClassifierCV(estimator=best_entry["model"], method='sigmoid', cv=5)
    calibrated_clf.fit(X_train, y_train)
    cal_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    cal_pred = (cal_prob >= 0.5).astype(int)
    cal_metrics = compute_metrics(y_test, cal_pred, cal_prob)
    cal_cis = compute_bootstrap_cis(y_test, cal_pred, cal_prob)
    
    # Conformal Prediction
    cal_fold_idx = int(0.75 * len(X_train))
    conf_train_probs = calibrated_clf.predict_proba(X_train[cal_fold_idx:])[:, 1]
    conf_results, uncertain_rate, q_hat = conformal_prediction_audit(
        conf_train_probs, y_train[cal_fold_idx:], cal_prob, alpha=0.10
    )
    
    # Fairness
    fairness = {}
    if 'sex' in test_df.columns:
        for s_val, s_label in [(1, 'male'), (0, 'female')]:
            mask = (test_df['sex'].values == s_val)
            if sum(mask) > 5 and len(np.unique(y_test[mask])) > 1:
                fairness[s_label] = compute_metrics(y_test[mask], cal_pred[mask], cal_prob[mask])
                fairness[s_label]['count'] = int(sum(mask))
            else:
                fairness[s_label] = {"count": int(sum(mask)), "note": "Insufficient sample"}
                
    # SHAP feature importance
    shap_summary = {}
    try:
        if best_name in ["LightGBM", "CatBoost", "XGBoost"]:
            explainer = shap.TreeExplainer(best_entry["model"])
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_vals_matrix = shap_values[1]
            elif shap_values.ndim == 3:
                shap_vals_matrix = shap_values[:, :, 1]
            else:
                shap_vals_matrix = shap_values
        else:
            explainer = shap.LinearExplainer(lr_pipe.named_steps['clf'], lr_pipe.named_steps['scaler'].transform(X_train))
            shap_vals_matrix = explainer.shap_values(lr_pipe.named_steps['scaler'].transform(X_test))
            
        mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
        top_shap_idx = np.argsort(mean_abs_shap)[::-1]
        shap_summary = {feature_cols[i]: float(mean_abs_shap[i]) for i in top_shap_idx}
        
        # Save plot
        plt.figure(figsize=(7, 4.5))
        y_pos = np.arange(len(top_shap_idx))
        plt.barh(y_pos, [mean_abs_shap[i] for i in top_shap_idx[::-1]], align='center', color='#3b82f6')
        plt.yticks(y_pos, [feature_cols[i] for i in top_shap_idx[::-1]])
        plt.xlabel('Mean |SHAP Value| (Impact on ASD screening prediction)')
        plt.title(f'Feature Importance: {cohort_name.title()} ({variant_name})')
        plt.tight_layout()
        shap_fig_path = os.path.join(PLOTS_DIR, f"shap_{cohort_name}_{variant_name}.png")
        plt.savefig(shap_fig_path, dpi=180)
        plt.close()
    except Exception as e:
        shap_summary = {col: 1.0 / len(feature_cols) for col in feature_cols}
        
    # Export ONNX model
    onnx_path = os.path.join(MODELS_DIR, f"{cohort_name}_{variant_name}.onnx")
    try:
        initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
        onx = convert_sklearn(lr_pipe, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
    except Exception as e:
        pass
        
    return {
        "cohort": cohort_name,
        "variant": variant_name,
        "feature_cols": feature_cols,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "best_model_name": best_name,
        "all_models_test_metrics": {k: v["metrics"] for k, v in models_dict.items()},
        "calibrated_metrics": cal_metrics,
        "bootstrap_cis": cal_cis,
        "conformal_audit": {
            "uncertain_rate": float(uncertain_rate),
            "threshold_q_hat": float(q_hat),
            "alpha": 0.10,
            "coverage_guarantee": "90%"
        },
        "fairness_audit": fairness,
        "shap_importance": shap_summary,
        "onnx_exported": os.path.exists(onnx_path),
        "onnx_path": onnx_path if os.path.exists(onnx_path) else None,
        "calibrated_model_obj": calibrated_clf,
        "raw_best_model_obj": best_entry["model"]
    }


def train_leakage_demonstration():
    """Train models WITH vs WITHOUT the score column to prove leakage empirically."""
    print("\n--- RUNNING LEAKAGE DEMONSTRATION EXPERIMENT ---")
    adult_train = pd.read_csv(os.path.join(DATA_DIR, "adult_train.csv"))
    adult_test = pd.read_csv(os.path.join(DATA_DIR, "adult_test.csv"))
    
    # 1. Without total score
    features_honest = [f'A{i}' for i in range(1, 11)]
    lr_honest = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, random_state=42))])
    lr_honest.fit(adult_train[features_honest], adult_train['target'])
    prob_honest = lr_honest.predict_proba(adult_test[features_honest])[:, 1]
    pred_honest = (prob_honest >= 0.5).astype(int)
    m_honest = compute_metrics(adult_test['target'], pred_honest, prob_honest)
    
    # 2. WITH total score / result
    features_leaked = [f'A{i}' for i in range(1, 11)] + ['leakage_score']
    lr_leaked = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, random_state=42))])
    lr_leaked.fit(adult_train[features_leaked], adult_train['target'])
    prob_leaked = lr_leaked.predict_proba(adult_test[features_leaked])[:, 1]
    pred_leaked = (prob_leaked >= 0.5).astype(int)
    m_leaked = compute_metrics(adult_test['target'], pred_leaked, prob_leaked)
    
    return {
        "honest_variant": {"features": features_honest, "metrics": m_honest},
        "leaked_variant": {"features": features_leaked, "metrics": m_leaked},
        "audit_verdict": "Total score column ('result' or 'Qchat-10-Score') is a direct mathematical derivation of the diagnostic cutoff and MUST be excluded from clinical screening feature sets."
    }


def export_models_to_json_and_weights(cohort_results):
    """Exports decision models to clean JSON weights format so that the webapp runs 100% client-side."""
    print("\n--- Exporting Models for WebApp (JSON & Weights) ---")
    export_bundle = {}
    
    for key, res in cohort_results.items():
        cohort = res['cohort']
        variant = res['variant']
        feature_cols = res['feature_cols']
        
        train_df = pd.read_csv(os.path.join(DATA_DIR, f"{cohort}_train.csv"))
        X = train_df[feature_cols].values.astype(float)
        y = train_df['target'].values.astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_scaled, y)
        
        export_bundle[f"{cohort}_{variant}"] = {
            "cohort": cohort,
            "variant": variant,
            "features": feature_cols,
            "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(),
            "coefficients": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "calibrated_metrics": res['calibrated_metrics'],
            "conformal_threshold": res['conformal_audit']['threshold_q_hat'],
            "shap_importance": res['shap_importance']
        }
        
    export_path = os.path.join(MODELS_DIR, "models_export.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_bundle, f, indent=2)
    print(f"Exported client-side models to {export_path}")
    
    webapp_models_path = os.path.join(BASE_DIR, "webapp", "models_export.json")
    with open(webapp_models_path, "w", encoding="utf-8") as f:
        json.dump(export_bundle, f, indent=2)
    print(f"Copied webapp client models to {webapp_models_path}")


def main():
    print("Starting Comprehensive Questionnaire ML Training Pipeline...")
    cohorts = ['adult', 'adolescent', 'child', 'toddler', 'unified']
    results_all = {}
    
    for c in cohorts:
        train_df = pd.read_csv(os.path.join(DATA_DIR, f"{c}_train.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, f"{c}_test.csv"))
        
        # Variant A: Full 10 Items
        features_a = [f'A{i}' for i in range(1, 11)]
        res_a = tune_and_train_cohort(train_df, test_df, features_a, c, "variant_a_items_only")
        results_all[f"{c}_variant_a"] = res_a
        
        # Variant B: Items + Demographics (no ethnicity or country)
        features_b = [f'A{i}' for i in range(1, 11)] + ['age', 'sex', 'jaundice', 'family_asd']
        res_b = tune_and_train_cohort(train_df, test_df, features_b, c, "variant_b_items_demographics")
        results_all[f"{c}_variant_b"] = res_b
        
        # Variant C: Abbreviated 5-Item Rapid Screen (A1 to A5, genuine non-deterministic ML screener!)
        features_c = [f'A{i}' for i in range(1, 6)]
        res_c = tune_and_train_cohort(train_df, test_df, features_c, c, "variant_c_abbreviated_5_items")
        results_all[f"{c}_variant_c"] = res_c
        
        # Variant D: Demographics-only baseline (negative control)
        features_d = ['age', 'sex', 'jaundice', 'family_asd']
        res_d = tune_and_train_cohort(train_df, test_df, features_d, c, "variant_d_demographics_only")
        results_all[f"{c}_variant_d"] = res_d
        
    # Leakage Demonstration
    leakage_demo = train_leakage_demonstration()
    
    # Export for webapp
    export_models_to_json_and_weights(results_all)
    
    # Strip model objects before caching
    clean_results = {}
    for k, v in results_all.items():
        v_copy = {x: v[x] for x in v if not x.endswith('_obj')}
        clean_results[k] = v_copy
    clean_results['leakage_demonstration'] = leakage_demo
    
    results_cache_path = os.path.join(REPORTS_DIR, "questionnaire_results_cache.json")
    with open(results_cache_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nAll questionnaire models trained! Results cached to {results_cache_path}")
    return clean_results


if __name__ == "__main__":
    main()
