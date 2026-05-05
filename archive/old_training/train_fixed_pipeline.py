"""
Train Random Forest with FIXED data pipeline.

Key improvements over the original:
1. Uses master Behavioral Features file (correct data source)
2. Per-person z-scoring (removes 14x inter-person SCL variation)
3. Missingness indicators instead of injecting noise via median imputation
4. Derived ratio features (CharactersRatio, ErrorKeyRatio)
5. Proper imputation: fills NaN with 0 AFTER per-person z-scoring
   (0 = "this person's average" — far better than global median)

Also runs ablation experiments to quantify each fix's impact.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader_v2 import (
    load_and_merge_data, get_loso_splits,
    PHYSIOLOGY_FEATURES, COMPUTER_FEATURES, DERIVED_FEATURES,
    ALL_FEATURES, LABEL_COL, GROUP_COL,
)


def get_feature_list(add_derived=True, add_missingness=True):
    """Get the list of feature columns to use."""
    feats = list(ALL_FEATURES)
    if add_derived:
        feats += list(DERIVED_FEATURES)
    if add_missingness:
        feats += [f"{c}_missing" for c in PHYSIOLOGY_FEATURES]
    return feats


def run_experiment(df, feature_cols, experiment_name, label_col=LABEL_COL):
    """Run LOSO CV with Random Forest and return results."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Rows: {len(df)}, Participants: {df['PP'].nunique()}")

    fold_results = []

    for train_idx, test_idx, pp_id in get_loso_splits(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        X_train = train[feature_cols].values.astype(float)
        y_train = train[label_col].values
        X_test = test[feature_cols].values.astype(float)
        y_test = test[label_col].values

        # Skip if test set has only one class
        if len(np.unique(y_test)) < 2:
            print(f"  {pp_id}: skipped (single class in test)")
            continue

        # Impute remaining NaN with 0 (after per-person z-scoring, 0 = person's mean)
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale (mild effect after z-scoring, but ensures consistency)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train
        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        fold_results.append({
            "PP": pp_id, "Accuracy": acc, "F1_Macro": f1,
            "Precision": prec, "Recall": rec, "AUC_ROC": auc,
        })
        print(f"  {pp_id}: acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

    # Summary
    results_df = pd.DataFrame(fold_results)
    print(f"\n--- {experiment_name} SUMMARY ---")
    for metric in ["Accuracy", "F1_Macro", "AUC_ROC", "Precision", "Recall"]:
        vals = results_df[metric].dropna()
        print(f"  {metric}: {vals.mean():.4f} ± {vals.std():.4f}")

    return results_df


def main():
    print("=" * 60)
    print("FIXED DATA PIPELINE — RANDOM FOREST EXPERIMENTS")
    print("=" * 60)

    # ── Experiment 1: Original approach (for comparison baseline) ──
    # Load WITHOUT per-person z-scoring, WITHOUT missingness indicators
    print("\n>>> Loading data WITHOUT fixes (original approach)...")
    df_orig = load_and_merge_data(
        per_person_zscore=False,
        add_missingness=False,
        add_derived=False,
        use_behavioral_xlsx=True,
    )
    feats_orig = list(ALL_FEATURES)
    res_orig = run_experiment(df_orig, feats_orig, "1. ORIGINAL (no fixes)")

    # ── Experiment 2: Per-person z-scoring only ───────────────────
    print("\n>>> Loading data WITH per-person z-scoring...")
    df_zscore = load_and_merge_data(
        per_person_zscore=True,
        add_missingness=False,
        add_derived=False,
        use_behavioral_xlsx=True,
    )
    feats_zscore = list(ALL_FEATURES)
    res_zscore = run_experiment(df_zscore, feats_zscore, "2. PER-PERSON Z-SCORE")

    # ── Experiment 3: Z-score + missingness indicators ────────────
    print("\n>>> Loading data WITH z-scoring + missingness indicators...")
    df_miss = load_and_merge_data(
        per_person_zscore=True,
        add_missingness=True,
        add_derived=False,
        use_behavioral_xlsx=True,
    )
    feats_miss = list(ALL_FEATURES) + [f"{c}_missing" for c in PHYSIOLOGY_FEATURES]
    res_miss = run_experiment(df_miss, feats_miss, "3. Z-SCORE + MISSINGNESS")

    # ── Experiment 4: All fixes (z-score + missingness + derived) ─
    print("\n>>> Loading data WITH ALL fixes...")
    df_all = load_and_merge_data(
        per_person_zscore=True,
        add_missingness=True,
        add_derived=True,
        use_behavioral_xlsx=True,
    )
    feats_all = get_feature_list(add_derived=True, add_missingness=True)
    res_all = run_experiment(df_all, feats_all, "4. ALL FIXES")

    # ── Experiment 5: All fixes + drop fully-missing-physio participants
    print("\n>>> Loading data WITH ALL fixes + drop bad participants...")
    df_clean = load_and_merge_data(
        per_person_zscore=True,
        add_missingness=True,
        add_derived=True,
        use_behavioral_xlsx=True,
        drop_fully_missing_physio=True,
    )
    feats_clean = get_feature_list(add_derived=True, add_missingness=True)
    res_clean = run_experiment(df_clean, feats_clean, "5. ALL FIXES + CLEAN PARTICIPANTS")

    # ── Experiment 6: Computer-only with per-person z-scoring ─────
    print("\n>>> Computer-only features with per-person z-scoring...")
    df_comp = load_and_merge_data(
        per_person_zscore=True,
        add_missingness=False,
        add_derived=True,
        use_behavioral_xlsx=True,
    )
    feats_comp = list(COMPUTER_FEATURES) + list(DERIVED_FEATURES)
    res_comp = run_experiment(df_comp, feats_comp, "6. COMPUTER-ONLY + Z-SCORE")

    # ── Final comparison ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    experiments = [
        ("1. Original (no fixes)", res_orig),
        ("2. Per-person z-score", res_zscore),
        ("3. Z-score + missingness", res_miss),
        ("4. All fixes", res_all),
        ("5. All + clean participants", res_clean),
        ("6. Computer-only + z-score", res_comp),
    ]
    print(f"\n{'Experiment':<35} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 65)
    for name, res in experiments:
        acc = res["Accuracy"].mean()
        f1 = res["F1_Macro"].mean()
        auc = res["AUC_ROC"].dropna().mean()
        print(f"{name:<35} {acc:>8.4f} {f1:>8.4f} {auc:>8.4f}")

    # Save results
    os.makedirs("results/fixed_pipeline", exist_ok=True)
    for name, res in experiments:
        safe_name = name.split(". ")[1].replace(" ", "_").replace("+", "").lower()
        res.to_csv(f"results/fixed_pipeline/{safe_name}.csv", index=False)
    print("\nResults saved to results/fixed_pipeline/")


if __name__ == "__main__":
    main()
