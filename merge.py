import pandas as pd

derek = pd.read_csv("data/15-sample.csv")

raw = pd.read_csv("data/dataset_scin_cases.csv")

merged_df = pd.merge(derek, raw, on="image_id")
columns_to_keep = [
    "case_id",
    "image_id",
    "condition",
    "fitz_scale",
]
filtered_df = merged_df[columns_to_keep]

filtered_df.to_csv("data/output.csv", index=False)
