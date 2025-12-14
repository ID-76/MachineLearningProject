import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    mean_absolute_error,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*fill_value.*")

# Use relative paths to Data folder
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
DATA_PATH = os.path.join(DATA_DIR, "US_Accidents_processed_for_modeling.csv")
SUBSET_PATH = os.path.join(DATA_DIR, "US_Accidents_model_subset.csv")
USE_SUBSET_IF_AVAILABLE = True
ON_THE_FLY_SUBSET_FRAC = 0.15
SUBSET_STRATIFY_COL = "Severity"
MAX_CARDINALITY_BEFORE_DROP = 50
TARGETS = ["Severity"]


def columns_to_drop_for_target(X_cols: pd.Index, target: str) -> list:
    """Drop ultra-high-cardinality columns that don't provide useful signal."""
    drops = []
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

def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """Evaluate model with custom decision threshold for class 4."""
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Adjust predictions for class 4 based on threshold
    # If probability of class 4 > threshold, predict class 4
    class4_mask = y_proba[:, 3] > threshold
    y_pred[class4_mask] = 3
    
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    n_sev3 = np.sum(y_test == 2)
    n_sev4 = np.sum(y_test == 3)
    n_severe = n_sev3 + n_sev4
    severe_recall = (recall_per_class[2] * n_sev3 + recall_per_class[3] * n_sev4) / n_severe if n_severe > 0 else 0
    
    return {
        'recall_sev3': recall_per_class[2],
        'recall_sev4': recall_per_class[3],
        'severe_recall': severe_recall,
        'y_pred': y_pred
    }

def test_smote_only(X_train, X_test, y_train, y_test, label_encoder, model_type: str):
    """Test SMOTE oversampling only (no threshold tuning)."""
    print(f"\n{'='*60}")
    print(f"SMOTE TESTING: {model_type}")
    print(f"{'='*60}")
    
    results = {}
    
    # Use optimal params from earlier testing
    if model_type in ["ID3", "CART"]:
        max_depth, min_samples_leaf = 10, 20
    else:  # Random Forest
        max_depth, min_samples_leaf = 15, 50
    
    # 1. Baseline (no SMOTE)
    print("\n1. BASELINE (No SMOTE)")
    if model_type == "ID3":
        model = DecisionTreeClassifier(
            criterion="entropy", random_state=42,
            class_weight={0: 0, 1: 1, 2: 5, 3: 10},
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )
    elif model_type == "CART":
        model = DecisionTreeClassifier(
            criterion="gini", random_state=42,
            class_weight={0: 0, 1: 1, 2: 5, 3: 10},
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight={0: 0, 1: 1, 2: 5, 3: 10},
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )
    
    model.fit(X_train, y_train)
    baseline = evaluate_with_threshold(model, X_test, y_test, threshold=0.5)
    results['baseline'] = baseline
    print(f"  Recall(Sev3): {baseline['recall_sev3']:.4f}")
    print(f"  Recall(Sev4): {baseline['recall_sev4']:.4f}")
    print(f"  Severe Combined: {baseline['severe_recall']:.4f}")
    
    # 2. With SMOTE
    print("\n2. WITH SMOTE (oversample class 3 & 4)")
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"  Original class distribution: {np.bincount(y_train)}")
        print(f"  After SMOTE: {np.bincount(y_train_smote)}")
        
        if model_type == "ID3":
            model_smote = DecisionTreeClassifier(
                criterion="entropy", random_state=42,
                class_weight={0: 0, 1: 1, 2: 5, 3: 10},
                max_depth=max_depth, min_samples_leaf=min_samples_leaf
            )
        elif model_type == "CART":
            model_smote = DecisionTreeClassifier(
                criterion="gini", random_state=42,
                class_weight={0: 0, 1: 1, 2: 5, 3: 10},
                max_depth=max_depth, min_samples_leaf=min_samples_leaf
            )
        else:
            model_smote = RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
                class_weight={0: 0, 1: 1, 2: 5, 3: 10},
                max_depth=max_depth, min_samples_leaf=min_samples_leaf
            )
        
        model_smote.fit(X_train_smote, y_train_smote)
        smote_result = evaluate_with_threshold(model_smote, X_test, y_test, threshold=0.5)
        results['smote'] = smote_result
        print(f"  Recall(Sev3): {smote_result['recall_sev3']:.4f}")
        print(f"  Recall(Sev4): {smote_result['recall_sev4']:.4f}")
        print(f"  Severe Combined: {smote_result['severe_recall']:.4f}")
    except Exception as e:
        print(f"  SMOTE failed: {e}")
        results['smote'] = None
    
    return results

def run_decision_trees_for_target(data: pd.DataFrame, target: str):
    """Train and evaluate decision tree models for a given target."""
    if target not in data.columns:
        print(f"Skipping target '{target}': not found in data.")
        return

    X = data.drop(columns=[target])
    y = data[target]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Drop high-cardinality and problematic columns
    leak_drops = columns_to_drop_for_target(X.columns, target)
    if leak_drops:
        X = X.drop(columns=leak_drops)

    X = drop_high_cardinality_cats(X, threshold=MAX_CARDINALITY_BEFORE_DROP)

    # One-hot encode
    X_enc = pd.get_dummies(X, sparse=False)
    print(f"  Feature matrix shape after encoding: {X_enc.shape}")

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, train_size=0.8, random_state=42, stratify=y_enc
    )

    print(f"\n{'='*60}")
    print(f"Target: '{target}' - Decision Trees")
    print(f"{'='*60}")
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    
    # Test SMOTE for all models
    print("\n" + "="*70)
    print("TESTING SMOTE")
    print("="*70)
    
    id3_results = test_smote_only(X_train, X_test, y_train, y_test, label_encoder, "ID3")
    cart_results = test_smote_only(X_train, X_test, y_train, y_test, label_encoder, "CART")
    rf_results = test_smote_only(X_train, X_test, y_train, y_test, label_encoder, "Random Forest")
    
    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for model_name, results in [("ID3", id3_results), ("CART", cart_results), ("Random Forest", rf_results)]:
        print(f"\n{model_name}:")
        print(f"  Baseline:              Severe Recall={results['baseline']['severe_recall']:.4f}, Sev4={results['baseline']['recall_sev4']:.4f}")
        if results.get('smote'):
            print(f"  With SMOTE:            Severe Recall={results['smote']['severe_recall']:.4f}, Sev4={results['smote']['recall_sev4']:.4f}")
        
        # Find best approach
        best_name = "Baseline"
        best_recall = results['baseline']['severe_recall']
        if results.get('smote') and results['smote']['severe_recall'] > best_recall:
            best_name = "SMOTE"
            best_recall = results['smote']['severe_recall']
        
        print(f"  ⭐ BEST: {best_name} with {best_recall:.4f}")


    # Compare against individual SMOTE results
    print("\nComparison vs individual SMOTE-trained models:")
    print(f"  ID3 (SMOTE):  Severe={id3_results['smote']['severe_recall']:.4f}, Sev3={id3_results['smote']['recall_sev3']:.4f}, Sev4={id3_results['smote']['recall_sev4']:.4f}")
    print(f"  CART (SMOTE): Severe={cart_results['smote']['severe_recall']:.4f}, Sev3={cart_results['smote']['recall_sev3']:.4f}, Sev4={cart_results['smote']['recall_sev4']:.4f}")
    print(f"  RF (SMOTE):   Severe={rf_results['smote']['severe_recall']:.4f}, Sev3={rf_results['smote']['recall_sev3']:.4f}, Sev4={rf_results['smote']['recall_sev4']:.4f}")
    best_individual = max(
        [('ID3', id3_results['smote']['severe_recall']), ('CART', cart_results['smote']['severe_recall']), ('RF', rf_results['smote']['severe_recall'])],
        key=lambda x: x[1]
    )
    print(f"  ⭐ BEST INDIVIDUAL: {best_individual[0]} with {best_individual[1]:.4f}")

def main():
    """Load data and run decision tree models for all targets."""
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
            tmp = data.copy()
            strat_col = SUBSET_STRATIFY_COL if SUBSET_STRATIFY_COL in tmp.columns else None
            if strat_col is not None:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - frac, random_state=42)
                X = tmp
                y = tmp[strat_col]
                for keep_idx, _ in sss.split(X, y):
                    data = data.iloc[keep_idx].copy()
                print(f"Downsampled in-memory (stratified by {strat_col}) to shape={data.shape}")
            else:
                data = data.sample(frac=frac, random_state=42).copy()
                print(f"Downsampled in-memory (random) to shape={data.shape}")
    
    for target in TARGETS:
        run_decision_trees_for_target(data, target)

if __name__ == "__main__":
    main()