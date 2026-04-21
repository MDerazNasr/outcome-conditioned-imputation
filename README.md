# outcome-conditioned-imputation

A prototype pipeline comparing standard clinical data imputation against 
outcome-conditioned imputation on MIMIC-IV data. The core question is does 
knowing a patient's diagnosis improve reconstruction of their missing lab 
and vital values?

This is the first step toward a formal research contribution on 
outcome-conditioned reconstruction of missing clinical modalities.

## Research Direction

Standard imputation estimates P(X_missing | X_observed).
This prototype tests whether conditioning on the known diagnosis,
P(X_missing | X_observed, Diagnosis), reduces reconstruction error.

## Concepts

Three clinical concepts extracted from MIMIC-IV:

| Concept | Source | Clinical Association |
|---|---|---|
| Troponin | labevents | Acute MI |
| WBC | labevents | Sepsis |
| Blood Pressure | chartevents | Heart Failure |

## Diagnosis Groups

Assigned from ICD codes in diagnoses_icd.csv:

| Diagnosis | ICD Prefixes |
|---|---|
| Sepsis | A41, A40 |
| Acute MI | I21, I22 |
| Heart Failure | I50 |
| Pneumonia | J18, J15, J13 |

## Data

Built on the MIMIC-IV demo dataset (100 patients), publicly available 
at physionet.org/content/mimic-iv-demo without credentialing.

Troponin missingness is natural from the data (69%).
WBC and blood pressure missingness is simulated at 30% using a fixed 
random seed (99) to enable the imputation comparison.

Full MIMIC-IV experiments pending PhysioNet credentialed access approval.


## Setup

```bash
pip install -r requirements.txt
```

Place the MIMIC-IV demo files in the following structure:

data/
└── raw/
├── hosp/
│   ├── labevents.csv
│   └── diagnoses_icd.csv
└── icu/
└── chartevents.csv


## Running the Pipeline

Run scripts in order:

```bash
python scripts/extract_cohort.py
python scripts/missingness.py
python scripts/standard_imputation.py
python scripts/conditioned_imputation.py
python scripts/compare.py
```
## Outputs

| File | Description |
|---|---|
| outputs/cohort.csv | 100 patient cohort with missingness |
| outputs/missingness_report.txt | Missingness rates per concept and diagnosis group |
| outputs/standard_results.csv | Standard imputation error metrics |
| outputs/conditioned_results.csv | Conditioned imputation error metrics |
| outputs/comparison.csv | Side by side comparison table |
| outputs/comparison.txt | Plain text comparison with summary |

---

## Results

| Concept | Standard MAE | Conditioned MAE |
|---|---|---|
| Troponin | 0.178 | 0.199 |
| WBC | 4.464 | 5.067 |
| Blood Pressure | 23.385 | 24.692 |

Standard imputation outperforms naive stratified conditioning on the 
100 patient demo cohort. This is expected: splitting 100 patients into 
diagnosis groups leaves 7 to 9 patients per group, which is insufficient 
for MICE to learn within-group patterns. The same pipeline on full 
MIMIC-IV with thousands of patients per group is required to properly 
test the direction.

---

## Limitations

- Demo dataset only (100 patients)
- Group sizes of 7 to 9 patients are too small for MICE
- Imaging concepts deferred pending MIMIC-CXR access
- Conditioning mechanism is naive stratification
  (more principled approach to be determined with supervisor)
