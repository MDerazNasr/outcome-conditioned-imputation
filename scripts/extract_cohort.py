import pandas as pd
import numpy as np
import os


# paths
# setting up folder ref so every line in the script
# can say RAW instead of typing out full path everytime
RAW = "data/raw"
OUT = "outputs"
# create outputs folders
os.makedirs(OUT, exist_ok=True)

# Load raw tables
lab = pd.read_csv(f"{RAW}/hosp/labevents.csv")
diagnoses = pd.read_csv(f"{RAW}/hosp/diagnoses_icd.csv.gz")
chart = pd.read_csv(f"{RAW}/icu/chartevents.csv.gz")

# Define concept items ids
# every lav test and vital sign in MIMIC has a numeric ID
# Troponin has 2 Ids because MIMIC records two trop. variants, WBC has one
# Blood pressure has 2 because MIMIC records systolic differently depending on
# whether patient is on a monitor or manual mesaurement
# IDs are standard across all MIMIC-IV datasets
TROPONIN_IDS = [51003, 51002]
WBC_IDS = [51301]
BP_IDS = [220179, 220050]

# Extract Lab Concepts
# filters the lab table to only rows matching item IDs
# sorts by time,and takes the first recorded value per patient
# you take the first value because you want the admission baseline, not a later measurement that may already reflect treatment. It returns a two column dataframe: subject_id and the concept value.


def extract_first_lab(lab_df, item_ids, concept_name):
    filtered = lab_df[lab_df["itemid"].isin(item_ids)].copy()
    filtered = filtered.sort_values("charttime")
    first = filtered.groupby("subject_id")["valuenum"].first().reset_index()
    first.columns = ["subject_id", concept_name]
    return first


troponin = extract_first_lab(lab, TROPONIN_IDS, "troponin")
wbc = extract_first_lab(lab, WBC_IDS, "wbc")

# Extract vitals concept
# same as lab concepts, but read from charevents because blood pressure is a vital sign not a lab result


def extract_first_chart(chart_df, item_ids, concept_name):
    filtered = chart_df[chart_df["itemid"].isin(item_ids)].copy()
    filtered = filtered.sort_values("charttime")
    first = filtered.groupby("subject_id")["valuenum"].first().reset_index()
    first.columns = ["subject_id", concept_name]
    return first


bp = extract_first_chart(chart, BP_IDS, "blood_pressure")

# Extract diagnosis groups
'''
This is the diagnosis assignment. 
ICD codes in MIMIC are strings like "A4101" or "I5020". 
You only need the first three characters to identify the condition category.
The function loops through every patient, checks whether any of their ICD codes match your diagnosis map prefixes, and assigns the first match.
If no match is found the patient gets labeled "other".
Those patients will still be in the cohort but won't be used for outcome conditioning.
'''
DIAGNOSIS_MAP = {
    "sepsis": ["A41", "A40"],
    "acute_mi": ["I21", "I22"],
    "heart_failure": ["I50"],
    "pneumonia": ["J18", "J15", "J13"]
}


def assign_diagnosis(diag_df, diagnosis_map):
    diag_df = diag_df.copy()
    diag_df["icd_prefix"] = diag_df["icd_code"].str[:3]
    results = []
    for subject_id, group in diag_df.groupby("subject_id"):
        assigned = "other"
        for label, prefixes in diagnosis_map.items():
            if group["icd_prefix"].isin(prefixes).any():
                assigned = label
                break
        results.append({"subject_id": subject_id, "diagnosis": assigned})
    return pd.DataFrame(results)


diagnosis = assign_diagnosis(diagnoses, DIAGNOSIS_MAP)

# Merge into one cohort
# how="left" means every patient stays in the cohort even if they habe no tropnin value
# This is how missing values appear naturally through in the output
cohort = diagnosis.copy()
cohort = cohort.merge(troponin, on="subject_id", how="left")
cohort = cohort.merge(wbc, on="subject_id", how="left")
cohort = cohort.merge(bp, on="subject_id", how="left")

# Save into output and prints summary
cohort.to_csv(f"{OUT}/cohort.csv", index=False)
print(f"Cohort saved: {len(cohort)} patients")
print(cohort["diagnosis"].value_counts())
print(cohort.head())
print(cohort[["troponin", "wbc", "blood_pressure"]].isnull().sum())

rng = np.random.default_rng(42)

for concept in ["wbc", "blood_pressure"]:
    mask = rng.random(len(cohort)) < 0.3
    cohort.loc[mask, concept] = np.nan

cohort.to_csv(f"{OUT}/cohort.csv", index=False)
print("\nAfter simulated missingness:")
print(cohort[["troponin", "wbc", "blood_pressure"]].isnull().sum())
