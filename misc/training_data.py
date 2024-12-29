import pandas as pd
import ast

conditions = {
    "Eczema": 1,
    "Impetigo": 2,
    "Urticaria": 3,
    "Psoriasis": 4,
    "Tinea": 5,
    "Allergic Contact Dermatitis": 6,
}

df = pd.read_csv("../data/dataset_scin_labels.csv")
output_rows = []

for index, row in df.iterrows():
    max_weight = 0
    max_condition = ""

    # converting this to a dictionary
    try:
        dictionary = ast.literal_eval(row["weighted_skin_condition_label"])
    except (ValueError, SyntaxError):
        print(f"Error parsing row {index}: {row['weighted_skin_condition_label']}")
        continue

    for condition, class_num in conditions.items():
        if condition in dictionary and dictionary[condition] > max_weight:
            max_weight = dictionary[condition]
            max_condition = condition

    if max_condition:
        output_rows.append(
            {
                "case_id": row["case_id"],
                "label": f"Class{conditions[max_condition]}",
                "split": "Valid",
            }
        )

output_df = pd.DataFrame(output_rows)
path_df = pd.read_csv("../data/dataset_scin_cases.csv")

merged_df = pd.merge(output_df, path_df, on="case_id", how="left")

selected_columns = merged_df[["image_id", "label", "split"]]
# rename image_id -> image_path
selected_columns = selected_columns.rename(columns={"image_id": "image_path"})

selected_columns.to_csv("../data/data.csv", index=False)
