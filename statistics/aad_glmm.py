import numpy as np
import pandas as pd
from pymer4.models import glmer

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
# Replace with your own file path
df = pd.read_csv("aad_trial_level_results.csv")

# Expected columns:
#   subject   : subject ID (string)
#   group_HI  : 0 = NH, 1 = HI
#   correct   : 0 = wrong classification, 1 = correct classification

required_cols = {"subject", "group_HI", "correct"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --------------------------------------------------
# 2. Basic cleaning / type checks
# --------------------------------------------------
df = df.copy()

# Drop rows with missing values in key columns
df = df.dropna(subset=["subject", "group_HI", "correct"])

# Enforce types
df["subject"] = df["subject"].astype(str)
df["group_HI"] = df["group_HI"].astype(int)
df["correct"] = df["correct"].astype(int)

# Sanity checks
if not set(df["group_HI"].unique()).issubset({0, 1}):
    raise ValueError("group_HI must contain only 0 (NH) and 1 (HI).")

if not set(df["correct"].unique()).issubset({0, 1}):
    raise ValueError("correct must contain only 0 and 1.")

# Optional: print quick overview
n_subjects_total = df["subject"].nunique()
n_subjects_nh = df.loc[df["group_HI"] == 0, "subject"].nunique()
n_subjects_hi = df.loc[df["group_HI"] == 1, "subject"].nunique()

print("\nDATA OVERVIEW")
print("-------------")
print(f"Total rows (windows/trials): {len(df)}")
print(f"Total subjects: {n_subjects_total}")
print(f"NH subjects: {n_subjects_nh}")
print(f"HI subjects: {n_subjects_hi}")
print("\nMean raw accuracy by group:")
print(df.groupby("group_HI")["correct"].mean().rename({0: "NH", 1: "HI"}))

# --------------------------------------------------
# 3. Fit mixed-effects logistic regression
# --------------------------------------------------
# Model:
#   correct ~ group_HI + (1 | subject)
#
# Interpretation:
#   Intercept = log-odds of correct decoding for NH (group_HI = 0)
#   group_HI  = change in log-odds for HI relative to NH

model = glmer(
    "correct ~ group_HI + (1 | subject)",
    data=df,
    family="binomial"
)

# Fit model
# exponentiate=False keeps coefficients on log-odds scale
model.fit(exponentiate=False, summary=True)

print("\nMODEL SUMMARY")
print("-------------")
print(model.result_fit)

# --------------------------------------------------
# 4. Extract fixed effects table
# --------------------------------------------------
# pymer4 stores the fitted summary table in result_fit
# We'll try to extract the row for group_HI
fixed_effects = model.result_fit

print("\nFIXED EFFECTS TABLE")
print("-------------------")
print(fixed_effects)

# --------------------------------------------------
# 5. Compute odds ratio for the group effect
# --------------------------------------------------
# We expect a row named "group_HI"
# Depending on pymer4 version, column names are usually lower-case-ish
# We'll search robustly.

term_col = None
for c in fixed_effects.columns:
    if str(c).lower() in {"term", "effect", "name"}:
        term_col = c
        break

if term_col is None:
    raise ValueError("Could not find the term/effect column in model.result_fit.")

group_row = fixed_effects[fixed_effects[term_col].astype(str) == "group_HI"]
if len(group_row) != 1:
    raise ValueError("Could not uniquely identify the group_HI row in fixed effects table.")

group_row = group_row.iloc[0]

# Try common column names
estimate_col = next((c for c in fixed_effects.columns if str(c).lower() == "estimate"), None)
se_col = next((c for c in fixed_effects.columns if str(c).lower() in {"se", "std.error", "std_error"}), None)
z_col = next((c for c in fixed_effects.columns if "z" in str(c).lower()), None)
p_col = next((c for c in fixed_effects.columns if str(c).lower() in {"p", "p-val", "pvalue", "p_value"}), None)
ci_low_col = next((c for c in fixed_effects.columns if "ci-low" in str(c).lower() or "2.5" in str(c).lower()), None)
ci_high_col = next((c for c in fixed_effects.columns if "ci-high" in str(c).lower() or "97.5" in str(c).lower()), None)

beta = float(group_row[estimate_col])
odds_ratio = np.exp(beta)

print("\nPRIMARY GROUP EFFECT")
print("--------------------")
print(f"log-odds coefficient for HI vs NH: {beta:.4f}")
print(f"odds ratio for HI vs NH: {odds_ratio:.4f}")

if se_col is not None:
    print(f"SE: {float(group_row[se_col]):.4f}")

if z_col is not None:
    print(f"z-statistic: {float(group_row[z_col]):.4f}")

if p_col is not None:
    print(f"p-value: {group_row[p_col]}")

if ci_low_col is not None and ci_high_col is not None:
    ci_low = float(group_row[ci_low_col])
    ci_high = float(group_row[ci_high_col])
    print(f"95% CI (log-odds): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"95% CI (odds ratio): [{np.exp(ci_low):.4f}, {np.exp(ci_high):.4f}]")

# --------------------------------------------------
# 6. Predicted probabilities for NH and HI
# --------------------------------------------------
# These are easier to report than log-odds.
pred_df = model.empredict({"group_HI": [0, 1]})

print("\nPREDICTED PROBABILITIES")
print("-----------------------")
print(pred_df)

# --------------------------------------------------
# 7. Save outputs for reporting
# --------------------------------------------------
fixed_effects.to_csv("glmm_fixed_effects.csv", index=False)
pred_df.to_csv("glmm_predicted_probabilities.csv", index=False)

# Optional: subject-level raw accuracies for descriptive plotting
subject_acc = (
    df.groupby(["subject", "group_HI"], as_index=False)["correct"]
      .mean()
      .rename(columns={"correct": "mean_accuracy"})
)
subject_acc.to_csv("subject_level_mean_accuracy.csv", index=False)

print("\nSaved:")
print("  glmm_fixed_effects.csv")
print("  glmm_predicted_probabilities.csv")
print("  subject_level_mean_accuracy.csv")