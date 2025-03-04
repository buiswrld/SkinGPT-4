import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from misc.data_gen.add_fitz import add_fitzpatrick_ratings

conditions = {
    "Eczema": 1,
    "Impetigo": 2,
    "Urticaria": 3,
    "Psoriasis": 4,
    "Tinea": 5,
    "Allergic Contact Dermatitis": 6,
}

### ADJUST THIS
MIN_CONFIDENCE_THRESHOLD = 0.6
###

OMIT_EDGE_CASES = False

df = pd.read_csv("../../data/raw/dataset_scin_labels.csv")
output_rows = []

for index, row in df.iterrows():
    try:
        dictionary = ast.literal_eval(row["weighted_skin_condition_label"])
    except (ValueError, SyntaxError):
        print(f"Error parsing row {index}: {row['weighted_skin_condition_label']}")
        continue

    max_weight = 0
    max_condition = ""
    total_weight = sum(dictionary.values())
    weight_counts = {}

    for condition, weight in dictionary.items():
        if weight > max_weight:
            max_weight = weight
            max_condition = condition
        if OMIT_EDGE_CASES:
            weight_counts[weight] = weight_counts.get(weight, 0) + 1

    confidence = max_weight / total_weight if total_weight > 0 else 0
    no_edge = True
    if confidence >= MIN_CONFIDENCE_THRESHOLD and max_condition in conditions:
        if OMIT_EDGE_CASES and weight_counts[max_weight] != 1:
            no_edge = False
        if no_edge:
            output_rows.append(
                {
                    "case_id": row["case_id"],
                    "label": f"{max_condition}",
                }
            )
            no_edge = True

output_df = pd.DataFrame(output_rows)
path_df = pd.read_csv("../../data/raw/dataset_scin_cases.csv")
merged_df = pd.merge(output_df, path_df, on="case_id", how="left")
selected_columns = merged_df[["image_id", "label"]]
selected_columns = selected_columns.rename(columns={"image_id": "image_path"})

final_df = add_fitzpatrick_ratings(selected_columns, "../../data/raw/dataset_scin_labels.csv", "../data/raw/dataset_scin_cases.csv")

final_df.drop_duplicates(inplace=True)

train_df, temp_df = train_test_split(final_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df])
final_df["image_path"] = final_df["image_path"].str.replace("dataset/images/", "images/")

final_df.to_csv(f"../../data/training/fitz/fitz_{int(MIN_CONFIDENCE_THRESHOLD*100)}c.csv", index=False)