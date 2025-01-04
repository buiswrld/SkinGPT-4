import pandas as pd

def convert_fitzpatrick_scale(fitz_value):
    if pd.isna(fitz_value):
        return None
    return int(fitz_value.replace("FST", ""))

def normalize_image_path(image_path):
    return image_path.split('/')[-1]

def add_fitzpatrick_ratings(data_df_path, labels_df_path, cases_df_path):
    data_df = pd.read_csv(data_df_path)
    labels_df = pd.read_csv(labels_df_path)
    cases_df = pd.read_csv(cases_df_path)

    labels_df = labels_df[['case_id', 'dermatologist_fitzpatrick_skin_type_label_1', 'dermatologist_fitzpatrick_skin_type_label_2', 'dermatologist_fitzpatrick_skin_type_label_3']]

    labels_df['fitz_1'] = labels_df['dermatologist_fitzpatrick_skin_type_label_1'].apply(convert_fitzpatrick_scale)
    labels_df['fitz_2'] = labels_df['dermatologist_fitzpatrick_skin_type_label_2'].apply(convert_fitzpatrick_scale)
    labels_df['fitz_3'] = labels_df['dermatologist_fitzpatrick_skin_type_label_3'].apply(convert_fitzpatrick_scale)

    labels_df['fitz'] = labels_df[['fitz_1', 'fitz_2', 'fitz_3']].mean(axis=1)

    cases_df = cases_df[['case_id', 'image_id']]
    cases_df['image_id'] = cases_df['image_id'].apply(normalize_image_path)
    data_df['image_path'] = data_df['image_path'].apply(normalize_image_path)

    merged_df = pd.merge(labels_df, cases_df, on='case_id')

    print("Merged DataFrame (labels_df + cases_df):")
    print(merged_df.head())

    final_df = pd.merge(data_df, merged_df, left_on='image_path', right_on='image_id', how='left')

    print("Final DataFrame before dropping columns:")
    print(final_df.head())

    final_df = final_df.drop(columns=['case_id', 'image_id', 'dermatologist_fitzpatrick_skin_type_label_1', 'dermatologist_fitzpatrick_skin_type_label_2', 'dermatologist_fitzpatrick_skin_type_label_3', 'fitz_1', 'fitz_2', 'fitz_3'])
    return final_df