import pandas as pd
from sklearn.model_selection import train_test_split
import itertools



def create_split(df, labels, filename):
    subset_df = df[df['label'].isin(labels)]
    train_df, temp_df = train_test_split(subset_df, test_size=0.3, stratify=subset_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, val_df, test_df])

    final_df.to_csv(filename, index=False)

def create_pairs(set_pairs):
    for idx, pair in enumerate(set_pairs, start=1):
        filename = f"../data/training/fitz/{conf_int}c_fitz_pairs/{idx}_data_{conf_int}c_subset_{pair[0]}_{pair[1]}.csv"
        create_split(df, set_pairs, filename)

def create_trips(set_trips):
    for idx, triplet in enumerate(set_trips, start=1):
        filename = f"../data/training/fitz/{conf_int}c_fitz_trips/5_data_{conf_int}c_subset_{triplet[0]}_{triplet[1]}_{triplet[2]}.csv"
        create_split(df, triplet, filename)

if __name__ == "__main__":

    ### ADJUST THIS
    CONFIDENCE = 0.60
    ###

    conf_int = int(CONFIDENCE * 100)
    df = pd.read_csv(f"../data/training/fitz_60c.csv")
    unique_labels = df['label'].unique()

    all_pairs = list(itertools.combinations(unique_labels, 2))
    all_triplets = list(itertools.combinations(unique_labels, 3))

    custom_pairs = (
        ("Eczema", "Allergic Contact Dermatitis"),
    )

    custom_trips = (
        ("Psoriasis", "Eczema", "Tinea"),
        ("Tinea", "Allergic Contact Dermatitis", "Psoriasis"),
        ("Impetigo", "Allergic Contact Dermatitis", "Psoriasis")
    )

    ### ADJUST THIS
    create_pairs(all_pairs)
    ###