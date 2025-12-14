import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, mean_absolute_error,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# Suppress FutureWarning about sparse dtype fill_value
warnings.filterwarnings("ignore", category=FutureWarning, message=".*fill_value.*")

# Use relative paths to Data folder
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
DATA_PATH = os.path.join(DATA_DIR, "US_Accidents_processed_for_modeling.csv")
SUBSET_PATH = os.path.join(DATA_DIR, "US_Accidents_model_subset.csv")
USE_SUBSET_IF_AVAILABLE = True
ON_THE_FLY_SUBSET_FRAC = 0.15  # if subset file missing, downsample in-memory; reduced to avoid OOM on one-hot encoding
SUBSET_STRATIFY_COL = "Severity"  # primary label to preserve distribution
MAX_CARDINALITY_BEFORE_DROP = 50  # drop categorical columns with >50 unique values before one-hot
# Target: Severity (ordinal 1-4 representing accident severity)
TARGETS = ["Severity"]

def create_derived_targets(df: pd.DataFrame) -> pd.DataFrame:
    """No additional targets need to be derived; Severity already exists in data."""
    return df.copy()

def columns_to_drop_for_target(X_cols: pd.Index, target: str) -> list:
    """Drop ultra-high-cardinality columns that don't provide useful signal."""
    drops = []
    # Always drop ultra-high-cardinality if present
    for c in ["Description", "Street", "Zipcode", "City", "County", "ID", "Weather_Timestamp", "Source"]:
        if c in X_cols:
            drops.append(c)
    return list(sorted(set(drops)))

def drop_high_cardinality_cats(X: pd.DataFrame, threshold: int = MAX_CARDINALITY_BEFORE_DROP) -> pd.DataFrame:
    """Drop object columns with cardinality > threshold to reduce one-hot explosion."""
    obj_cols = X.select_dtypes(include=['object']).columns
    high_card = [c for c in obj_cols if X[c].nunique() > threshold]
    if high_card:
        X = X.drop(columns=high_card)
        print(f"  Dropped high-cardinality columns (>{threshold} uniques): {high_card}")
    return X

def run_svm_variant(X_train, X_test, y_train, y_test, label_encoder, target: str, model_name: str, class_weight=None, C=1.0):
    """Train and evaluate SVM variant with specified class weights and C regularization.
    
    Args:
        class_weight: Class weight dict or 'balanced'
        C: Regularization strength (default 1.0). Higher = stricter fit, lower = more regularization
    """
    # Use specified class_weight, default to 'balanced' if None
    cw = class_weight if class_weight is not None else "balanced"
    
    # Pipeline: scale (sparse-safe) + linear SVM
    model = make_pipeline(
        StandardScaler(with_mean=False, with_std=True),
        LinearSVC(random_state=42, max_iter=3000, class_weight=cw, dual=False, C=C)
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Per-class recall
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    # PRIMARY METRICS - Focus on recall for severe classes
    recall_sev3 = recall_per_class[2]  # Class 3 (index 2)
    recall_sev4 = recall_per_class[3]  # Class 4 (index 3)
    
    # Severe recall: weighted average of class 3 and 4 recalls
    # Count class 3 and 4 samples in test set
    n_sev3 = np.sum(y_test == 2)
    n_sev4 = np.sum(y_test == 3)
    n_severe = n_sev3 + n_sev4
    severe_recall = (recall_sev3 * n_sev3 + recall_sev4 * n_sev4) / n_severe if n_severe > 0 else 0
    
    # Macro recall / Balanced accuracy
    macro_recall = np.mean(recall_per_class)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # SECONDARY METRICS
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # PR-AUC for class 4 (One-vs-Rest)
    # Need probability predictions; use decision_function as proxy
    try:
        y_scores = model.decision_function(X_test)
        # For OvR, binarize labels and get score for class 4
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        if y_scores.ndim == 1:  # Binary case
            pr_auc_class4 = 0.0
        else:
            pr_auc_class4 = average_precision_score(y_test_bin[:, 3], y_scores[:, 3])
    except:
        pr_auc_class4 = 0.0
    
    # ORDINAL / COST-AWARE METRICS
    # MAE on severity labels (convert back to original 1-4 scale)
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    
    # Quadratic weighted Cohen's Kappa
    qwk = cohen_kappa_score(y_test_original, y_pred_original, weights='quadratic')
    
    # Print results
    print(f"\n{'-'*70}")
    print(f"Model: {model_name}")
    print(f"{'-'*70}")
    
    # Organized metrics output - no repetition
    metrics_data = {
        "PRIMARY METRICS (Recall-focused)": {
            "Recall (Severity 3)": f"{recall_sev3:.4f}",
            "Recall (Severity 4)": f"{recall_sev4:.4f}",
            "Severe Recall (3+4 combined)": f"{severe_recall:.4f}",
            "Macro Recall": f"{macro_recall:.4f}",
            "Balanced Accuracy": f"{balanced_acc:.4f}",
        },
        "SECONDARY METRICS": {
            "Macro F1": f"{f1_macro:.4f}",
            "Matthews Corr Coef (MCC)": f"{mcc:.4f}",
            "PR-AUC (class 4, OvR)": f"{pr_auc_class4:.4f}",
        },
        "ORDINAL / COST-AWARE METRICS": {
            "MAE (severity scale)": f"{mae:.4f}",
            "Quadratic Weighted Kappa": f"{qwk:.4f}",
        }
    }
    
    for section_name, metrics in metrics_data.items():
        print(f"\n{section_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:<35} {value:>10}")
    
    print(f"\n{'='*70}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n{'='*70}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_], zero_division=0))

def run_svm_for_target(data: pd.DataFrame, target: str):
    data = create_derived_targets(data)
    if target not in data.columns:
        print(f"Skipping target '{target}': not found in data.")
        return

    X = data.drop(columns=[target])
    y = data[target]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Per-target leakage prevention
    leak_drops = columns_to_drop_for_target(X.columns, target)
    if leak_drops:
        X = X.drop(columns=leak_drops)

    # Drop high-cardinality categorical columns to prevent one-hot explosion
    X = drop_high_cardinality_cats(X, threshold=MAX_CARDINALITY_BEFORE_DROP)

    # One-hot encode with sparse matrices to save memory
    X_enc = pd.get_dummies(X, sparse=True)
    print(f"  Feature matrix shape after encoding: {X_enc.shape}")
    
    # Convert to dense numpy array to avoid sklearn conversion warnings
    X_enc = X_enc.to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, train_size=0.8, random_state=42, stratify=y_enc
    )

    print(f"\n{'='*60}")
    print(f"Target: '{target}' - Final Severity Model")
    print(f"{'='*60}")
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    
    # Final chosen configuration: aggressive weights favoring classes 3 and 4,
    # but still giving a small weight to class 1 (least severe)
    optimal_weights = {0: 0, 1: 1, 2: 100, 3: 300}
    run_svm_variant(
        X_train, X_test, y_train, y_test, label_encoder, target,
        "LinearSVC (final: weights 0/1/100/300, C=1.0)",
        class_weight=optimal_weights, C=1.0
    )

def main():
    # Prefer prebuilt subset when available
    if USE_SUBSET_IF_AVAILABLE and os.path.exists(SUBSET_PATH):
        data = pd.read_csv(SUBSET_PATH)
        print(f"Loaded stratified subset: {SUBSET_PATH} shape={data.shape}")
    else:
        data = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset: {DATA_PATH} shape={data.shape}")
        # Optional in-memory stratified downsample to speed-up experiments
        frac = float(ON_THE_FLY_SUBSET_FRAC)
        if 0 < frac < 1:
            tmp = create_derived_targets(data)
            strat_col = SUBSET_STRATIFY_COL if SUBSET_STRATIFY_COL in tmp.columns else None
            if strat_col is None:
                # try weather-derived fallback
                if "is_precip" in tmp.columns:
                    strat_col = "is_precip"
                elif "WeatherSimple" in tmp.columns:
                    strat_col = "WeatherSimple"
            if strat_col is not None:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - frac, random_state=42)
                X = tmp
                y = tmp[strat_col]
                for keep_idx, _ in sss.split(X, y):
                    data = data.iloc[keep_idx].copy()
                print(f"Downsampled in-memory (stratified by {strat_col}) to shape={data.shape}")
            else:
                # fallback random sample if no stratify label
                data = data.sample(frac=frac, random_state=42).copy()
                print(f"Downsampled in-memory (random) to shape={data.shape}")
    for target in TARGETS:
        run_svm_for_target(data, target)

if __name__ == "__main__":
    main()