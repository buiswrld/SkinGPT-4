import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import os


### ADJUST THIS
CONFIDENCE = 0.10
###

conf_int = int(CONFIDENCE * 100)
df = pd.read_csv(f"../../data/training/fitz/fitz_{int(CONFIDENCE*100)}c.csv")
unique_labels = df['label'].unique()

def create_split(df, labels, filename):
    # Filter the DataFrame for the specific labels
    subset_df = df[df['label'].isin(labels)]
    
    # Ensure there are enough samples for stratified splitting
    if subset_df.empty:
        print(f"Skipping {labels}: no samples found.")
        return
    if len(subset_df['label'].value_counts()) < len(labels):
        print(f"Skipping {labels}: not all labels have samples.")
        return

    try:
        # Split the subset into train, validation, and test sets
        train_df, temp_df = train_test_split(
            subset_df, test_size=0.3, stratify=subset_df['label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
        )
    except ValueError as e:
        print(f"Error processing {labels}: {e}")
        return

    # Assign splits and combine into a single DataFrame
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, val_df, test_df])
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    final_df.to_csv(filename, index=False)
    print(f"Saved file: {filename}")

def create_pairs(set_pairs):
    for idx, pair in enumerate(set_pairs, start=1):
        filename = f"../../data/training/fitz/{conf_int}c_fitz_pairs/{idx}_data_{conf_int}c_subset_{pair[0]}_{pair[1]}.csv"
        create_split(df, pair, filename)

def create_trips(set_trips):
    for idx, triplet in enumerate(set_trips, start=1):
        filename = f"../../data/training/fitz/{conf_int}c_fitz_trips/{idx}_data_{conf_int}c_subset_{triplet[0]}_{triplet[1]}_{triplet[2]}.csv"
        create_split(df, triplet, filename)

if __name__ == "__main__":

    # Calculate confidence interval and define file path
    conf_int = int(CONFIDENCE * 100)
    file_path = f"../../data/training/fitz/fitz_{conf_int}c.csv"

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the dataset
    df = pd.read_csv(file_path)

    # Validate that the dataset is not empty
    if df.empty:
        raise ValueError(f"The file {file_path} is empty!")

    # Get unique labels
    unique_labels = df['label'].unique()
    print("Available labels:", unique_labels)

    # Generate all pairs and triplets
    all_pairs = list(itertools.combinations(unique_labels, 2))
    all_triplets = list(itertools.combinations(unique_labels, 3))

    # Define custom pairs and triplets (optional)
    custom_pairs = [
        ("Eczema", "Allergic Contact Dermatitis"),
    ]
    custom_trips = [
        ("Psoriasis", "Eczema", "Tinea"),
        ("Tinea", "Allergic Contact Dermatitis", "Psoriasis"),
        ("Impetigo", "Allergic Contact Dermatitis", "Psoriasis")
    ]

    create_pairs(custom_pairs)
