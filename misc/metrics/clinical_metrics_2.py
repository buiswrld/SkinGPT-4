import pandas as pd
import numpy as np
import csv
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

def load_csv_data(csv_path):
    """
    Loads the evaluation data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(csv_path)

def initialize_arrays(num_samples):
    """
    Initializes arrays for storing predictions and skin tone data.

    Args:
        num_samples (int): Number of samples.

    Returns:
        dict: Dictionary containing initialized arrays.
    """
    return {
        "y_true": np.ones(num_samples, dtype=int),
        "y_pred_correct": np.zeros(num_samples, dtype=int),
        "y_pred_informative": np.zeros(num_samples, dtype=int),
        "y_pred_helpful": np.zeros(num_samples, dtype=int),
        "y_pred_understand": np.zeros(num_samples, dtype=int),
        "skin_tone": np.zeros(num_samples, dtype=int),
    }

def update_predictions(df, arrays):
    """
    Updates prediction arrays based on CSV data.
    """
    column_mapping = {
        "y_pred_correct": "SkinGPT-4's diagnosis is correct/relevant? (Yes/No)",
        "y_pred_informative": "SkinGPT-4's description is informative (Yes/No)",
        "y_pred_helpful": "SkinGPT-4 can help doctors with diagnosis (Yes/No)",
        "y_pred_understand": "SkinGPT-4 can help patients understand their disease better Yes/No)"
    }
    
    for key, column_name in column_mapping.items():
        arrays[key][: len(df)] = df[column_name].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
        df[key] = arrays[key]  # Add the prediction columns to the DataFrame

def calculate_fairness_metrics(y_true, y_pred, sensitive_features, description):
    """
    Calculates fairness metrics.
    """
    metrics = {"accuracy": accuracy_score}
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    accuracy = mf.overall["accuracy"]
    print(f"{description} -> Accuracy: {accuracy:.2f}")
    return accuracy, mf.difference(method="between_groups"), mf.difference(method="to_overall")

def main():
    evaluations_csv_path = "../../data/clinical_inference/evaluations.csv"
    sample_csv_path = "../../data/clinical_inference/10-sample.csv"
    
    df = load_csv_data(evaluations_csv_path)
    print("Column names in the DataFrame:", df.columns)  # Print column names
    
    # Load the 10-sample.csv file
    sample_df = load_csv_data(sample_csv_path)
    
    # Create a mapping from Internal Identifier to Skin Tone
    skin_tone_mapping = sample_df.set_index('image_path')['fitz'].to_dict()
    
    # Map Skin Tone to the main DataFrame
    df['Skin Tone'] = df['Internal Identifier'].map(skin_tone_mapping).fillna(0).astype(int)
    
    num_samples = len(df)
    arrays = initialize_arrays(num_samples)
    update_predictions(df, arrays)
    
    # Add y_true column to the DataFrame
    df['y_true'] = arrays['y_true']
    
    # Debugging: Print the first few rows of the DataFrame to verify the data
    print(df.head())
    
    unique_diseases = df["Condition"].unique()
    metrics = {}
    
    for disease in unique_diseases:
        disease_df = df[df["Condition"] == disease]
        # Debugging: Print the first few rows of the disease-specific DataFrame
        print(f"Data for {disease}:")
        print(disease_df.head())
        
        # Debugging: Print the y_true and y_pred_correct columns
        print(f"y_true for {disease}: {disease_df['y_true'].values}")
        print(f"y_pred_correct for {disease}: {disease_df['y_pred_correct'].values}")
        
        accuracy, _, _ = calculate_fairness_metrics(
            disease_df["y_true"], disease_df["y_pred_correct"], disease_df["Skin Tone"], f"{disease} (Correct)"
        )
        metrics[f"{disease} (Correct)"] = accuracy
    
    output_csv = "metrics_output.csv"
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, f"{value:.2f}"])
    print(f"Metrics have been written to {output_csv}")

if __name__ == "__main__":
    main()