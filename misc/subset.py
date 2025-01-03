import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/data.csv")
condition_counts = df['label'].value_counts()

def create_split(df, top_conditions, filename):
    subset_df = df[df['label'].isin(top_conditions)]
    train_df, temp_df = train_test_split(subset_df, test_size=0.3, stratify=subset_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, val_df, test_df])

    final_df.to_csv(filename, index=False)

# top 2 conditions subset
top_2_conditions = condition_counts.index[:2]
create_split(df, top_2_conditions, "../data/data_subset_top2.csv")

# top 3 conditions subset
top_3_conditions = condition_counts.index[:3]
create_split(df, top_3_conditions, "../data/data_subset_top3.csv")