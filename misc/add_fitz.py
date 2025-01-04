import pandas as pd

# Function to convert Fitzpatrick scale values to numerical values
def convert_fitzpatrick_scale(fitz_value):
    if pd.isna(fitz_value):
        return None
    return int(fitz_value.replace("FST", ""))

# Function to normalize image paths
def normalize_image_path(image_path):
    return image_path.split('/')[-1]

# Read the data files
data_df = pd.read_csv("data/training/data.csv")
labels_df = pd.read_csv("data/raw/dataset_scin_labels.csv")
cases_df = pd.read_csv("data/raw/dataset_scin_cases.csv")

# Extract relevant columns from labels_df
labels_df = labels_df[['case_id', 'dermatologist_fitzpatrick_skin_type_label_1', 'dermatologist_fitzpatrick_skin_type_label_2', 'dermatologist_fitzpatrick_skin_type_label_3']]

# Convert Fitzpatrick scale values to numerical values
labels_df['fitz_1'] = labels_df['dermatologist_fitzpatrick_skin_type_label_1'].apply(convert_fitzpatrick_scale)
labels_df['fitz_2'] = labels_df['dermatologist_fitzpatrick_skin_type_label_2'].apply(convert_fitzpatrick_scale)
labels_df['fitz_3'] = labels_df['dermatologist_fitzpatrick_skin_type_label_3'].apply(convert_fitzpatrick_scale)

# Calculate the average Fitzpatrick rating
labels_df['fitz'] = labels_df[['fitz_1', 'fitz_2', 'fitz_3']].mean(axis=1)

# Extract relevant columns from cases_df
cases_df = cases_df[['case_id', 'image_id']]

# Normalize image paths in cases_df
cases_df['image_id'] = cases_df['image_id'].apply(normalize_image_path)

# Normalize image paths in data_df
data_df['image_path'] = data_df['image_path'].apply(normalize_image_path)

# Merge labels_df with cases_df to get image IDs
merged_df = pd.merge(labels_df, cases_df, on='case_id')

# Debug: Check the merged_df
print("Merged DataFrame (labels_df + cases_df):")
print(merged_df.head())

# Merge the result with data_df to get image paths
final_df = pd.merge(data_df, merged_df, left_on='image_path', right_on='image_id', how='left')

# Debug: Check the final_df before dropping columns
print("Final DataFrame before dropping columns:")
print(final_df.head())

# Drop unnecessary columns
final_df = final_df.drop(columns=['case_id', 'image_id', 'dermatologist_fitzpatrick_skin_type_label_1', 'dermatologist_fitzpatrick_skin_type_label_2', 'dermatologist_fitzpatrick_skin_type_label_3', 'fitz_1', 'fitz_2', 'fitz_3'])

# Debug: Check the final_df after dropping columns
print("Final DataFrame after dropping columns:")
print(final_df.head())

# Save the updated DataFrame to the existing file
final_df.to_csv("data/training/fitz_data.csv", index=False)