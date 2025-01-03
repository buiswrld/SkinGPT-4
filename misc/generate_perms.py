import pandas as pd
from sklearn.model_selection import train_test_split
import itertools

df = pd.read_csv("../data/training/data.csv")
unique_labels = df['label'].unique()
label_pairs = list(itertools.combinations(unique_labels, 2))

def create_split(df, labels, filename):
    subset_df = df[df['label'].isin(labels)]
    train_df, temp_df = train_test_split(subset_df, test_size=0.3, stratify=subset_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, val_df, test_df])

    final_df.to_csv(filename, index=False)

for idx, pair in enumerate(label_pairs, start=1):
    filename = f"../data/training/subsets/{idx}_data_subset_{pair[0]}_{pair[1]}.csv"
    create_split(df, pair, filename)