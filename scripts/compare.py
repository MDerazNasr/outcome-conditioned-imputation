import pandas as pd    # for loading and manipulating results dataframes
import os

# paths
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# LOAD BOTH RESULTS FILES
# script 3 saved standard imputation error metrics to standard_results.csv
# script 4 saved conditioned imputation error metrics to conditioned_results.csv
# we load both and combine them into one comparison table

standard = pd.read_csv(f"{OUT}/standard_results.csv")
conditioned = pd.read_csv(f"{OUT}/conditioned_results.csv")

# COMBINE INTO ONE TABLE
# pd.concat stacks two dataframes on top of each other
# ignore_index=True resets the row numbers so they run 0 to 5
# instead of 0 to 2 twice

combined = pd.concat([standard, conditioned], ignore_index=True)

# BUILD THE SIDE BY SIDE COMPARISON
# pivot_table reshapes the data so that:
# rows = one per concept (troponin, wbc, blood_pressure)
# columns = one per method (standard, conditioned)
# values = mae and rmse
#
# this makes it easy to read the comparison at a glance

comparison = combined.pivot_table(
    index="concept",          # one row per concept
    columns="method",         # one column per method
    values=["mae", "rmse"]    # show both error metrics
).round(3)

# flatten the column names so they read cleanly
# instead of ("mae", "standard_imputation") we get "mae_standard"
comparison.columns = [
    f"{metric}_{method.replace('_imputation', '')}"
    for metric, method in comparison.columns
]

comparison = comparison.reset_index()

# COMPUTE IMPROVEMENT
# improvement shows how much the conditioned method changed error vs standard
# positive number means conditioned was worse
# negative number means conditioned was better
#
# formula: conditioned error minus standard error
# if conditioned MAE is lower the result is negative which means improvement

comparison["mae_improvement"] = (
    comparison["mae_conditioned"] - comparison["mae_standard"]
).round(3)

comparison["rmse_improvement"] = (
    comparison["rmse_conditioned"] - comparison["rmse_standard"]
).round(3)

# ADD A READABLE VERDICT COLUMN
# for each concept, state plainly whether conditioning helped or hurt
# this makes the table self-explanatory when you show it to Roy


def verdict(row):
    if row["mae_improvement"] < 0:
        return "conditioned better"
    elif row["mae_improvement"] > 0:
        return "standard better"
    else:
        return "no difference"


comparison["verdict"] = comparison.apply(verdict, axis=1)

# PRINT THE COMPARISON
print("=" * 70)
print("IMPUTATION COMPARISON: STANDARD VS OUTCOME-CONDITIONED")
print("=" * 70)
print(comparison.to_string(index=False))

# PRINT A PLAIN ENGLISH SUMMARY
# this section explains the numbers in words
# useful for the meeting with Roy

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for _, row in comparison.iterrows():
    concept = row["concept"]
    improvement = row["mae_improvement"]
    verdict_str = row["verdict"]

    if verdict_str == "conditioned better":
        direction = f"improved by {abs(improvement)} MAE units"
    elif verdict_str == "standard better":
        direction = f"was worse by {abs(improvement)} MAE units"
    else:
        direction = "showed no difference"

    print(f"{concept}: outcome conditioning {direction}")

print("\nNOTE: Results are on MIMIC-IV demo cohort (100 patients).")
print("Group sizes of 7-9 patients per diagnosis are too small for")
print("MICE to learn within-group patterns reliably.")
print("Full MIMIC-IV experiments required to properly evaluate the direction.")

# SAVE THE COMPARISON
comparison.to_csv(f"{OUT}/comparison.csv", index=False)

# write a plain text version for the meeting summary document
with open(f"{OUT}/comparison.txt", "w") as f:
    f.write("IMPUTATION COMPARISON: STANDARD VS OUTCOME-CONDITIONED\n")
    f.write("=" * 70 + "\n\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\nSUMMARY\n")
    f.write("=" * 70 + "\n")
    for _, row in comparison.iterrows():
        concept = row["concept"]
        improvement = row["mae_improvement"]
        verdict_str = row["verdict"]
        if verdict_str == "conditioned better":
            direction = f"improved by {abs(improvement)} MAE units"
        elif verdict_str == "standard better":
            direction = f"was worse by {abs(improvement)} MAE units"
        else:
            direction = "showed no difference"
        f.write(f"{concept}: outcome conditioning {direction}\n")
    f.write("\nNOTE: Results are on MIMIC-IV demo cohort (100 patients).\n")
    f.write("Group sizes of 7-9 patients per diagnosis are too small for\n")
    f.write("MICE to learn within-group patterns reliably.\n")
    f.write("Full MIMIC-IV experiments required to properly evaluate the direction.\n")

print(f"\nComparison saved to {OUT}/comparison.txt")
