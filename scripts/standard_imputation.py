import pandas as pd
import numpy as np
# miceforest is a fast implementation of MICE imputation
# MICE stands for Multiple Imputation by Chained Equations
# it estimates each missing value by building a model
# that predicts it from all other available columns
import miceforest as mf
import os

from scripts.missingness import CONCEPTS

# Path
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# Load Cohort
# load the cohort script 1 produced
# this has natural and simulated missigness already baked in
cohort = pd.read_csv(f"{OUT}/cohort.csv")

# Define concepts
# the three columns we are imputing
CONCEPTS = ["troponin", "wbc", "blood_pressure"]

# Seperate out the ground truth
'''
Doing this before we impute anything we need to save the real values
so we can measure how accurate the imputation was later

the problem: once we impute, the missing cells get filled in
and we lose track of which values were real and which were estimated

the solution: save a copy of the og cohort right now,
then after imputation we compare the imputed values against
the real values that were present in this original copy

'''

# saves the real values before any imputation
# ground_truth has NaN where data was missing
# and real values where data was present:
ground_truth = cohort[CONCEPTS].copy()
