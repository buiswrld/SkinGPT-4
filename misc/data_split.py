import pandas as pd
import ast
from sklearn.model_selection import train_test_split

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
                "label": f"{max_condition}",
            }
        )

output_df = pd.DataFrame(output_rows)
path_df = pd.read_csv("../data/dataset_scin_cases.csv")
merged_df = pd.merge(output_df, path_df, on="case_id", how="left")
selected_columns = merged_df[["image_id", "label"]]
selected_columns = selected_columns.rename(columns={"image_id": "image_path"})

train_df, temp_df = train_test_split(selected_columns, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df])

final_df.to_csv("../data/data.csv", index=False)