import argparse
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, equalized_odds_difference, demographic_parity_difference
from sklearn.metrics import accuracy_score

def calculate_fairness_metrics(csv_file):
    df = pd.read_csv(csv_file)

    true_labels = df['True Label'].values
    predicted_labels = df['Predicted Label'].values
    sensitive_attributes = df['Sensitive Attribute'].values

    metric_frame = MetricFrame(
        metrics={"accuracy": accuracy_score},
        y_true=true_labels,
        y_pred=predicted_labels,
        sensitive_features=sensitive_attributes
    )

    equalized_odds_diff = equalized_odds_difference(true_labels, predicted_labels, sensitive_attributes)
    demographic_parity_diff = demographic_parity_difference(true_labels, predicted_labels, sensitive_attributes)

    print(f"Equalized Odds Difference: {equalized_odds_diff}")
    print(f"Demographic Parity Difference: {demographic_parity_diff}")

    print("Detailed metrics:")
    print(metric_frame.by_group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fairness metrics for finetuned SkinGPT")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV containing preds")
    args = parser.parse_args()

    calculate_fairness_metrics(args.csv_file)