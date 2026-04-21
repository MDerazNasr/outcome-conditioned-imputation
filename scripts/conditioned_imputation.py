# imports
import pandas as pd        # for data manipulation
import numpy as np         # for numerical operations
import miceforest as mf    # for MICE imputation
import os

# paths
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# load cohort
cohort = pd.read_csv(f"{OUT}/cohort.csv")   # same cohort as script 3
# natural and simulated missingness intact

# define concepts
CONCEPTS = ["troponin", "wbc", "blood_pressure"]

# RECREATE THE EXACT SAME HOLDOUT AS SCRIPT 3
# this is critical
# we must withhold the exact same cells as script 3
# otherwise the comparison in script 5 is not fair
# same random seed 99 guarantees identical holdout rows

rng = np.random.default_rng(99)
holdout_mask = pd.DataFrame(False, index=cohort.index, columns=CONCEPTS)

for concept in CONCEPTS:
    present_idx = cohort[concept].dropna().index
    n_holdout = max(1, int(len(present_idx) * 0.2))
    chosen = rng.choice(present_idx, size=n_holdout, replace=False)
    holdout_mask.loc[chosen, concept] = True

# save ground truth before masking
ground_truth = cohort[CONCEPTS].copy()

# apply holdout mask
impute_df = cohort[CONCEPTS].copy()
impute_df[holdout_mask] = np.nan

# Identify which cells are missing in the data we feed to the imputer
# This includes original missing values AND our artificial holdout
missing_mask = impute_df.isnull()

# GET THE DIAGNOSIS GROUPS
# get all unique diagnosis labels
diagnosis_groups = cohort["diagnosis"].unique()

# IMPUTE WITHIN EACH DIAGNOSIS Group
# starts with NaN where holdout mask applied
conditioned_imputed = impute_df.copy()

for diagnosis in diagnosis_groups:

    # get the row indices for this diagnosis group
    group_idx = cohort[cohort["diagnosis"] == diagnosis].index

    # get the concept values for just this group
    group_data = impute_df.loc[group_idx, CONCEPTS].copy()

    # check if this group has any missing values to impute
    if not group_data.isnull().any().any():
        conditioned_imputed.loc[group_idx, CONCEPTS] = group_data
        continue

    # miceforest requires a minimum number of samples to build its models (LightGBM)
    # and to perform mean matching. Small groups often cause IndexError in mean matching.
    # We increase the threshold to 20 to ensure stability.
    if len(group_data) < 20:
        # fallback: use the group mean for imputation
        for concept in CONCEPTS:
            missing_in_group = group_data[concept].isnull()
            if missing_in_group.any():
                group_mean = group_data[concept].mean()
                if pd.isna(group_mean):
                    # if the whole group is missing this concept, fall back to global mean
                    group_mean = cohort[concept].mean()
                group_data.loc[missing_in_group, concept] = group_mean
        conditioned_imputed.loc[group_idx, CONCEPTS] = group_data
        continue

    # miceforest 6.0.5 requires a reset index starting from 0
    original_group_index = group_data.index
    group_data_reset = group_data.reset_index(drop=True)

    # run MICE on this diagnosis group only
    kernel = mf.ImputationKernel(
        data=group_data_reset,
        save_all_iterations_data=True,
        random_state=42
    )
    # Reduce iterations to 2 for smaller datasets to avoid potential overfit/stability issues
    kernel.mice(2)
    group_imputed = kernel.complete_data()

    # restore the original index so we can map back
    group_imputed.index = original_group_index

    # write back
    conditioned_imputed.loc[group_idx, CONCEPTS] = group_imputed

# MEASURE IMPUTATION ERROR ON HOLDOUT ROWS
results = []
for concept in CONCEPTS:
    was_held_out = holdout_mask[concept]
    real = ground_truth.loc[was_held_out, concept]
    pred = conditioned_imputed.loc[was_held_out, concept]
    mae = (real - pred).abs().mean().round(3)
    rmse = round(((real - pred) ** 2).mean() ** 0.5, 3)

    n_imputed = missing_mask[concept].sum()

    results.append({
        "concept": concept,
        "n_imputed": n_imputed,
        "mae": mae,
        "rmse": rmse,
        "method": "conditioned_imputation"
    })

# Save results
results_df = pd.DataFrame(results)
conditioned_cohort = cohort.copy()
conditioned_cohort[CONCEPTS] = conditioned_imputed
conditioned_cohort.to_csv(f"{OUT}/conditioned_imputed.csv", index=False)
results_df.to_csv(f"{OUT}/conditioned_results.csv", index=False)

# print summary
print("=" * 50)
print("OUTCOME-CONDITIONED IMPUTATION RESULTS")
print("=" * 50)
print(results_df.to_string(index=False))
