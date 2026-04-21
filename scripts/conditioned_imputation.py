# ── IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd        # for data manipulation
import numpy as np         # for numerical operations
import miceforest as mf    # for MICE imputation
import os

# ── PATHS ──────────────────────────────────────────────────────────────────
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ── LOAD COHORT ────────────────────────────────────────────────────────────
cohort = pd.read_csv(f"{OUT}/cohort.csv")   # same cohort as script 3
# natural and simulated missingness intact

# ── DEFINE CONCEPTS ────────────────────────────────────────────────────────
CONCEPTS = ["troponin", "wbc", "blood_pressure"]

# ── STEP 1: RECREATE THE EXACT SAME HOLDOUT AS SCRIPT 3 ───────────────────
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

# ── STEP 2: GET THE DIAGNOSIS GROUPS ──────────────────────────────────────
# this is what makes script 4 different from script 3
# instead of imputing on the full cohort ignoring diagnosis
# we split the cohort by diagnosis group first
# then impute within each group separately
#
# the intuition: a sepsis patient's missing WBC should be estimated
# using other sepsis patients, not a mixed population of ICU patients
# the known diagnosis constrains what the missing value probably looks like

# get all unique diagnosis labels
diagnosis_groups = cohort["diagnosis"].unique()
# other, heart_failure, sepsis, acute_mi, pneumonia

# ── STEP 3: IMPUTE WITHIN EACH DIAGNOSIS GROUP ────────────────────────────
# we will collect imputed values here
# start with a copy of the masked dataframe
# then fill in group by group

# starts with NaN where holdout mask applied
conditioned_imputed = impute_df.copy()
# we fill these in group by group below

for diagnosis in diagnosis_groups:

    # get the row indices for this diagnosis group
    group_idx = cohort[cohort["diagnosis"] == diagnosis].index

    # get the concept values for just this group
    group_data = impute_df.loc[group_idx, CONCEPTS].copy()

    # check if this group has enough patients to run MICE
    # MICE needs at least 2 patients to build a model
    # if a group is too small we fall back to standard imputation
    # this handles the pneumonia group which only has 2 patients

    if len(group_data) < 4:
        # fallback: use the group mean for imputation
        # this is still outcome conditioned because we are using
        # statistics from the same diagnosis group
        # it is just a simpler model than MICE
        for concept in CONCEPTS:
            missing_in_group = group_data[concept].isnull()
            if missing_in_group.any():
                group_mean = group_data[concept].mean()
                if pd.isna(group_mean):
                    # if the whole group is missing this concept
                    # fall back to global mean
                    group_mean = cohort[concept].mean()
                group_data.loc[missing_in_group, concept] = group_mean
        conditioned_imputed.loc[group_idx, CONCEPTS] = group_data
        continue   # move to next diagnosis group

    # check if this group has any missing values to impute
    # if nothing is missing in this group we skip MICE entirely
    if not group_data.isnull().any().any():
        conditioned_imputed.loc[group_idx, CONCEPTS] = group_data
        continue

    # run MICE on this diagnosis group only
    # same settings as script 3 for fair comparison
    kernel = mf.ImputationKernel(
        data=group_data,
        save_all_iterations=True,
        random_state=42
    )
    kernel.mice(3)
    group_imputed = kernel.complete_data()

    # write the imputed values back into the full results dataframe
    conditioned_imputed.loc[group_idx, CONCEPTS] = group_imputed

# ── STEP 4: MEASURE IMPUTATION ERROR ON HOLDOUT ROWS ─────────────────────
# identical measurement logic to script 3
# we measure only on the rows we deliberately withheld
# so the comparison against script 3 is direct and fair

results = []

for concept in CONCEPTS:
    was_held_out = holdout_mask[concept]
    real = ground_truth.loc[was_held_out, concept]
    pred = conditioned_imputed.loc[was_held_out, concept]
    mae = (real - pred).abs().mean().round(3)
    rmse = round(((real - pred) ** 2).mean() ** 0.5, 3)
    n_imputed = was_held_out.sum()

    results.append({
        "concept": concept,
        "n_imputed": n_imputed,
        "mae": mae,
        "rmse": rmse,
        "method": "conditioned_imputation"
    })

# ── STEP 5: SAVE RESULTS ───────────────────────────────────────────────────
results_df = pd.DataFrame(results)

# save the fully imputed cohort
conditioned_cohort = cohort.copy()
conditioned_cohort[CONCEPTS] = conditioned_imputed
conditioned_cohort.to_csv(f"{OUT}/conditioned_imputed.csv", index=False)

# save the error metrics
results_df.to_csv(f"{OUT}/conditioned_results.csv", index=False)

# print a readable summary
print("=" * 50)
print("OUTCOME-CONDITIONED IMPUTATION RESULTS")
print("=" * 50)
print(results_df.to_string(index=False))
