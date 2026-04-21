import pandas as pd
import os

# Paths
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# Load cohort
# load the CSV that script 1 produced
# this gives us one row per patient with
# columns: subject_id, diagnosis, troponin, wbc, blood_pressure
cohort = pd.read_csv(f"{OUT}/cohort.csv")


# Define the 3 concepts we are measuring
# we loop over this insetad of repeating the same code 3 times
CONCEPTS = ["troponin", "wbc", "blood_pressure"]


# Overall missingness rate per concept
# isnull() returns True for every cell that is missing and False for every cell that has a value
# mean() on a True/False column gives you the proportion of True values
# multiplying by 100 converts that proportion to a percentage
# round(1) rounds to one decimal place

overall = (
    cohort[CONCEPTS]       # select only the three concept columns
    .isnull()              # True where missing, False where present
    .mean()                # proportion missing per column
    .mul(100)              # convert to percentage
    .round(1)              # round to one decimal
    .reset_index()         # turn the column names into a regular column called "index"
    .rename(columns={      # rename the columns to something readable
        "index": "concept",
        0: "pct_missing"
    })
)

# Missingness rate per concept per diagnosis group
# groupby("diagnosis") splits the cohort into seperate groups, one per diagnosis
# then for each group we compute the same missingness percentage as above

group_missing = (
    cohort
    # for each diagnosis group, look at the concept columns
    .groupby("diagnosis")[CONCEPTS]
    .apply(lambda g: g.isnull()  # for each group g, compute missingness
           .mean()  # proportion missing per concept within that group
           .mul(100)  # convert to percentage
           .round(1))  # round to one decimal
    # flatten the grouped result back into a regular dataframe
    .reset_index(())


)


# Patient count per diagnosis group
# this tells us how many patients are in each group
# value_counts() counts how many times each diagnosis labels appears
# reset_index() turns it ito a two column dataframe

group_counts = (
    cohort["diagnosis"]
    .value_counts()
    .reset_index()
    .rename(columns={
        "diagnosis": "diagnosis",
        "count": "n_patients"
    })
)

# print the report
print("=" * 50)
print("MISSINGNESS REPORT")
print("=" * 50)

print("\nOVERALL MISSINGNESS RATE PER CONCEPT")
# to_string(index=False) prints the dataframe
print(overall.to_string(index=False))
# without the row numbers on the left side

print("\nPATIENT COUNT PER DIAGNOSIS GROUP")
print(group_counts.to_string(index=False))

print("\nMISSINGNESS RATE PER CONCEPT BY DIAGNOSIS GROUP")
print(group_missing.to_string(index=False))

# Save the report to file
# we write the same output to a text file so we have a record of it
# open() with "w" mode creates the file if it does not exist and overwrites it if it does

with open(f"{OUT}/missingness_report.txt", "w") as f:
    f.write("MISSINGNESS REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write("OVERALL MISSINGNESS RATE PER CONCEPT\n")
    f.write(overall.to_string(index=False))
    f.write("\n\nPATIENT COUNT PER DIAGNOSIS GROUP\n")
    f.write(group_counts.to_string(index=False))
    f.write("\n\nMISSINGNESS RATE PER CONCEPT BY DIAGNOSIS GROUP\n")
    f.write(group_missing.to_string(index=False))

print(f"\nReport saved to {OUT}/missingness_report.txt")
