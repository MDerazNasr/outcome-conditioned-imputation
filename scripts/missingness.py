import pandas as pd
import os

# Paths
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# Load cohort
# load the CSV that script 1 produced
cohort = pd.read_csv(f"{OUT}/cohort.csv")


# Define the 3 concepts we are measuring
# this gives us one row per patient with
# columns: subject_id, diagnosis, troponin, wbc, blood_pressure
CONCEPTS = ["troponin", "wbc", "blood_pressure"]
