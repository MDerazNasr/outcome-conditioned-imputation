import pandas as pd
import numpy as np
# miceforest is a fast implementation of MICE imputation
# MICE stands for Multiple Imputation by Chained Equations
# it estimates each missing value by building a model
# that predicts it from all other available columns
import miceforest as mf
import os

# Path
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# Load Cohort
# load the cohort script 1 produced
# this has natural and simulated missigness already baked in
cohort = pd.read_csv(f"{OUT}/cohort.csv")

# Define concepts
CONCEPTS = ["troponin", "wbc", "blood_pressure"]

# ── CREATE HOLDOUT SET ─────────────────────────────────────────────────────
# for each concept, take rows where the value IS present
# randomly withhold 20% of those rows
# impute them using MICE
# measure error on the withheld rows only

rng = np.random.default_rng(99)
holdout_mask = pd.DataFrame(False, index=cohort.index, columns=CONCEPTS)

for concept in CONCEPTS:
    present_idx = cohort[concept].dropna().index
    n_holdout = max(1, int(len(present_idx) * 0.2))
    chosen = rng.choice(present_idx, size=n_holdout, replace=False)
    holdout_mask.loc[chosen, concept] = True

# save ground truth before masking
ground_truth = cohort[CONCEPTS].copy()

# Prepare data for miceforest
# apply holdout: set chosen cells to NaN so MICE has to impute them
impute_df = cohort[CONCEPTS].copy()
impute_df[holdout_mask] = np.nan

# Identify which cells are missing in the data we feed to the imputer
# This includes original missing values AND our artificial holdout
missing_mask = impute_df.isnull()

# RUN MICE IMPUTATION
# miceforest.ImputationKernel() sets up the imputation engine
# data= is the dataframe with missing values
# save_all_iterations_data=True keeps all imputation rounds for inspection
# random_state=42 makes the result reproducible

kernel = mf.ImputationKernel(
    data=impute_df,
    save_all_iterations_data=True,
    random_state=42
)

# mice() runs the actual imputation
# iterations=3 means MICE refines its estimates 3 times
# more iterations = more accurate but slower
# 3 is standard for a prototype
kernel.mice(3)

# complete_data() returns the fully imputed dataframe
# every cell that NaN is now filled in with an estimated value
imputed = kernel.complete_data()  # same shape as impute_df but no missing values

# MEASURE IMPUTATION ERROR
# we only measure error on the HOLDOUT rows (cells we hid)
# because those are the only ones where we know the "true" value
# but the imputer thought they were missing.

results = []   # we will collect one row of results per concept

for concept in CONCEPTS:
    # was_holdout is True for rows we artificially masked
    was_holdout = holdout_mask[concept]

    # get the real values for rows where we hid the data
    real = ground_truth.loc[was_holdout, concept]

    # get the imputed values for the same rows
    pred = imputed.loc[was_holdout, concept]

    # mean absolute error: average of the absolute differences
    # lower is better
    mae = (real - pred).abs().mean().round(3)

    # root mean squared error: penalises large errors more than small ones
    # also lower is better
    rmse = ((real - pred) ** 2).mean() ** 0.5
    rmse = round(rmse, 3)

    # count how many cells were actually missing or held out for this concept
    n_imputed = missing_mask[concept].sum()

    results.append({
        "concept": concept,
        "n_imputed": n_imputed,
        "mae": mae,
        "rmse": rmse,
        "method": "standard_imputation"
    })

# save results
results_df = pd.DataFrame(results)

# save the fully imputed cohort so script 5 can use it
# We merge the imputed values back into the original cohort
imputed_cohort = cohort.copy()
imputed_cohort[CONCEPTS] = imputed

imputed_cohort.to_csv(f"{OUT}/standard_imputed.csv", index=False)

# save the error metrics
results_df.to_csv(f"{OUT}/standard_results.csv", index=False)

# print a readable summary
print("=" * 50)
print("STANDARD IMPUTATION RESULTS")
print("=" * 50)
print(results_df.to_string(index=False))

'''
Once MICE fills in the missing cells you cannot tell which values were real and which were estimated. You save the original copy first so you always have something to measure against. This is the standard evaluation pattern for any imputation experiment.
What MICE actually does
MICE does not use a single model. It builds one model per column, predicting each missing column from all other columns. It repeats this process across multiple iterations, each time improving its estimates using the updated values from the previous round. By iteration 3 the estimates have stabilized. This is why it is called chained equations: each column's model feeds into the next.
Why standard imputation ignores diagnosis deliberately
Dropping the diagnosis column before running MICE is not an oversight. It is the definition of standard imputation. No diagnosis information. No outcome conditioning. This is the baseline you are trying to beat in script 4.
'''
