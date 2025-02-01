from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score

import os
import pandas as pd
import numpy as np
from pandas import DataFrame

def initialize_arrays(total_images):
    """
    Initializes numpy arrays for true labels and predictions.

    Args:
        total_images (int): Total number of images.

    Returns:
        dict: Dictionary containing initialized arrays.
    """
    arrays = {
        "y_true": np.ones(total_images, dtype=int),
        "y_pred": np.zeros(total_images, dtype=int),
    }
    return arrays

def fill_arrays(arrays, inference_df: DataFrame) -> dict:
    """
    Fills prediction and truth arrays based on the inference DataFrame.

    Args:
        arrays (dict): Dictionary containing initialized arrays for predictions and true labels
        inference_df (DataFrame): DataFrame containing inference results and ground truth

    Returns:
        dict: Updated arrays dictionary with filled values
    """
    for index, row in inference_df.iterrows():
        arrays["y_pred"][index] = 1 if row["label"] == row["truth"] else 0
    return arrays

def get_skintones(inference_df: DataFrame) -> list[int]:
    """
    Extracts skin tone values from the inference DataFrame.

    Args:
        inference_df (DataFrame): DataFrame containing inference results with skin tone data

    Returns:
        list[int]: List of Fitzpatrick skin tone values rounded to integers
    """
    fitz_values = inference_df["fitz"].tolist()
    fitz_values = [5 if value == 6 else value for value in fitz_values]
    return fitz_values


def combine_csv(inference_csv, ground_truth_csv: str) -> DataFrame:
    """
    Combines inference results with ground truth data by merging on image path.

    Args:        inference_csv (str): Path to CSV file containing model inference results
        ground_truth_csv (str): Path to CSV file containing ground truth labels

    Returns:
        DataFrame: Combined DataFrame with both inference and ground truth data
    """
    inference_df = pd.read_csv(inference_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv)
    
    combined_df = pd.merge(inference_df, ground_truth_df, on='image_path')

    combined_df["fitz"] = combined_df["fitz"].round()

    return combined_df

def calculate_fairness_metrics(y_true, y_pred, sensitive_features, description):
    """
    Calculates and returns fairness metrics (between_groups, to_overall).

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        sensitive_features (np.ndarray): Sensitive features array.
        description (str): Description for logging.

    Returns:
        dict: between_groups
        dict: to_overall
    """
    try:
        metrics = {
            "false negative rate": false_negative_rate,
            "accuracy": accuracy_score,
        }

        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )

        print(f'metrics frame for {description}\n: {mf.overall}')
        print(f'metric frame by class: \n {mf.by_group}')

        between_groups = mf.difference(method='between_groups')
        to_overall = mf.difference(method='to_overall')

        print(f'{description} -> between_groups: {between_groups}, to_overall: {to_overall}')

        return mf, between_groups, to_overall
    except Exception as e:
        print(f"Failed to calculate fairness metrics for {description}: {e}")
        return None, None
    
def process_directory(directory_path):
    prediction_csv = os.path.join(directory_path, "prediction.csv")
    ground_truth_csv = os.path.join(directory_path, "truth.csv")

    if not (os.path.exists(prediction_csv) and os.path.exists(ground_truth_csv)):
        print(f"skipping {directory_path} - missing required CSV files")
        return

    combined_df = combine_csv(prediction_csv, ground_truth_csv)

    arrays = initialize_arrays(len(combined_df))
    arrays = fill_arrays(arrays, combined_df)

    sensitive_features = get_skintones(combined_df)

    mf, between_groups, to_overall = calculate_fairness_metrics(
        arrays["y_true"],
        arrays["y_pred"],
        sensitive_features,
        f"combined_{os.path.basename(directory_path)}"
    )

    with open(os.path.join(directory_path, "final_results.txt"), "w") as f:
        f.write(f"\nMetrics for {os.path.basename(directory_path)}:\n")
        f.write(f"Metrics frame overall:\n{mf.overall}\n")
        f.write(f"Metrics frame by group:\n{mf.by_group}\n")
        f.write(f"Between groups difference: {between_groups}\n")
        f.write(f"To overall difference: {to_overall}\n")
        f.write("-" * 80 + "\n")

    print("flushed metrics to csv")

def update(directory_path):
    """
    Process a specific directory path.

    Args:
        directory_path (str): The path to the directory to process.
    """
    if not os.path.isdir(directory_path):
        print(f"{directory_path} is not a valid directory")
        return

    print(f"processing directory: {directory_path}")
    process_directory(directory_path)
    

def main(ablations = False):
    if ablations:
        metrics_dir = "./data/metrics/ablations"
    else:
        metrics_dir = "./data/metrics"
    for subdir in os.listdir(metrics_dir):
        subdir_path = os.path.join(metrics_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
            
        print(f"processing directory: {subdir}")

        if ablations:
            for inner_subdir in os.listdir(subdir_path):
                inner_subdir_path = os.path.join(subdir_path, inner_subdir)

                if not os.path.isdir(inner_subdir_path):
                    continue

                print(f"processing inner directory: {inner_subdir}")

                process_directory(inner_subdir_path)
        else:
            process_directory(subdir_path)

if __name__ == "__main__":
    main(ablations=True)
