import gspread
import requests
import csv
import io
import sys
import pickle
import numpy as np
from google.auth import default
from google.colab import auth
from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score
from functools import partial


def authenticate_google():
    """
    Authenticates the user with Google credentials.

    Returns:
        gspread.Client: Authorized gspread client.
    """
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
    print("Authenticated and authorized with Google.")
    return gc


def fetch_google_sheet(gc, sheet_url, worksheet_index=0):
    """
    Fetches a Google Sheet and returns the specified worksheet.

    Args:
        gc (gspread.Client): Authorized gspread client.
        sheet_url (str): URL of the Google Sheet.
        worksheet_index (int): Index of the worksheet to fetch.

    Returns:
        gspread.Worksheet: Retrieved worksheet.
    """
    try:
        sheet = gc.open_by_url(sheet_url)
        worksheet = sheet.get_worksheet(worksheet_index)
        print(f"Fetched worksheet {worksheet_index} from {sheet_url}")
        return worksheet
    except Exception as e:
        print(f"Failed to fetch Google Sheet: {e}")
        sys.exit(1)


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
        "y_pred_correct": np.zeros(total_images, dtype=int),
        "y_pred_informative": np.zeros(total_images, dtype=int),
        "y_pred_helpful": np.zeros(total_images, dtype=int),
        "y_pred_understand": np.zeros(total_images, dtype=int),
        "skin_tone": np.zeros(total_images, dtype=int),
    }
    print("Initialized prediction and true label arrays.")
    return arrays


def update_predictions(conditions, y_pred, total_images):
    """
    Updates prediction arrays based on conditions.

    Args:
        conditions (list): List of condition values from Google Sheets.
        y_pred (np.ndarray): Prediction array to update.
        total_images (int): Total number of images.
    """
    y_pred[: len(conditions[:total_images])] = [
        1 if value.lower() == "yes" else 0 for value in conditions[:total_images]
    ]
    print(f"Updated predictions for {len(conditions[:total_images])} images.")


def fetch_csv_data(csv_url):
    """
    Fetches CSV data from a given URL.

    Args:
        csv_url (str): URL of the CSV file.

    Returns:
        list: List of CSV rows.
    """
    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        csv_data = response.content.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_data))
        next(reader)  # Skip header
        data = list(reader)
        print(f"Fetched and parsed CSV data from {csv_url}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch CSV data: {e}")
        sys.exit(1)


def map_skin_tone(ninth_column_values, csv_data, skin_tone_array, total_images):
    """
    Maps the ninth column values to skin tone using CSV data.

    Args:
        ninth_column_values (list): Values from the ninth column of Google Sheets.
        csv_data (list): List of CSV rows.
        skin_tone_array (np.ndarray): Array to store skin tone values.
        total_images (int): Total number of images.
    """
    lookup = {
        row[0]: int(row[2]) for row in csv_data if len(row) > 2 and row[2].isdigit()
    }
    for i, value in enumerate(ninth_column_values[:total_images]):
        for key in lookup:
            if value in key:  # Check if value is a substring of any key
                skin_tone_array[i] = lookup[key]
                break  # Stop searching once a match is found
        else:
            skin_tone_array[i] = 0  # Set to 0 if no match is found


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
            sensitive_features=sensitive_features,
        )

        print(f"metrics frame for {description}: {mf.overall}")

        between_groups = mf.difference(method="between_groups")
        to_overall = mf.difference(method="to_overall")

        print(
            f"{description} -> between_groups: {between_groups}, to_overall: {to_overall}"
        )

        return between_groups, to_overall
    except Exception as e:
        print(f"Failed to calculate fairness metrics for {description}: {e}")
        return None, None


def calculate_hallucination_rate(sheet, total_entries):
    """
    Calculates the hallucination rate from the Google Sheet.

    Args:
        sheet (gspread.Worksheet): Google Sheet worksheet.
        total_entries (int): Total number of entries to process.

    Returns:
        float: Hallucination rate.
    """
    try:
        hallucinations = sheet.col_values(8)[1 : total_entries + 1]
        counter = (
            hallucinations.count("yes")
            + hallucinations.count("Yes")
            + hallucinations.count("YES")
        )
        print(f"{counter} / {total_entries}")
        rate = counter / total_entries
        return rate
    except Exception as e:
        print(f"Failed to calculate hallucination rate: {e}")
        return 0.0


def main():
    # Configuration
    sheet_url = "https://docs.google.com/spreadsheets/d/16uBL9zzLeM-jvESowqEoRKxSkrUMVVNuP8ATXyIJBw0/edit?gid=1714650698#gid=1714650698"
    csv_url = (
        "https://raw.githubusercontent.com/buiswrld/SkinGPT-4/main/data/10-sample.csv"
    )
    total_images = 100
    split_index = 50

    gc = authenticate_google()
    worksheet = fetch_google_sheet(gc, sheet_url)

    arrays = initialize_arrays(total_images)

    diagnosis_correct = worksheet.col_values(4)[1:]
    diagnosis_informative = worksheet.col_values(5)[1:]
    diagnosis_helpful = worksheet.col_values(6)[1:]
    diagnosis_understand = worksheet.col_values(7)[1:]

    update_predictions(diagnosis_correct, arrays["y_pred_correct"], total_images)
    update_predictions(
        diagnosis_informative, arrays["y_pred_informative"], total_images
    )
    update_predictions(diagnosis_helpful, arrays["y_pred_helpful"], total_images)
    update_predictions(diagnosis_understand, arrays["y_pred_understand"], total_images)

    csv_data = fetch_csv_data(csv_url)
    ninth_column_values = worksheet.col_values(8)[1:]
    map_skin_tone(ninth_column_values, csv_data, arrays["skin_tone"], total_images)

    metrics = {}

    # Contact Dermatitis (Correct)
    (
        metrics["Contact Dermatitis (Correct) (Between Groups)"],
        metrics["Contact Dermatitis (Correct) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][:split_index],
        arrays["y_pred_correct"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Correct)",
    )

    # Eczema (Correct)
    (
        metrics["Eczema (Correct) (Between Groups)"],
        metrics["Eczema (Correct) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][split_index:total_images],
        arrays["y_pred_correct"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Correct)",
    )

    # Contact Dermatitis (Informative)
    (
        metrics["Contact Dermatitis (Informative) (Between Groups)"],
        metrics["Contact Dermatitis (Informative) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][:split_index],
        arrays["y_pred_informative"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Informative)",
    )

    # Eczema (Informative)
    (
        metrics["Eczema (Informative) (Between Groups)"],
        metrics["Eczema (Informative) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][split_index:total_images],
        arrays["y_pred_informative"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Informative)",
    )

    # Contact Dermatitis (Helpful)
    (
        metrics["Contact Dermatitis (Helpful) (Between Groups)"],
        metrics["Contact Dermatitis (Helpful) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][:split_index],
        arrays["y_pred_helpful"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Helpful)",
    )

    # Eczema (Helpful)
    (
        metrics["Eczema (Helpful) (Between Groups)"],
        metrics["Eczema (Helpful) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][split_index:total_images],
        arrays["y_pred_helpful"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Helpful)",
    )

    # Contact Dermatitis (Understand)
    (
        metrics["Contact Dermatitis (Understand) (Between Groups)"],
        metrics["Contact Dermatitis (Understand) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][:split_index],
        arrays["y_pred_understand"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Understand)",
    )

    # Eczema (Understand)
    (
        metrics["Eczema (Understand) (Between Groups)"],
        metrics["Eczema (Understand) (To Overall)"],
    ) = calculate_fairness_metrics(
        arrays["y_true"][split_index:total_images],
        arrays["y_pred_understand"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Understand)",
    )

    hallucination_sheet = "https://docs.google.com/spreadsheets/d/1O1jKNm-_KRsafYoSxS_sd-8rTaYlIce1FyYpcae_15g/edit?gid=1714650698#gid=1714650698"
    gc = authenticate_google()
    hallucination_worksheet = fetch_google_sheet(gc, hallucination_sheet)

    hallucination_rate = calculate_hallucination_rate(
        hallucination_worksheet, total_entries=298
    )
    metrics["Hallucination Rate"] = hallucination_rate
    print(f"Final Hallucinations Rate: {hallucination_rate:.3f}")

    print("Final metrics dictionary:\n", metrics)

    output_csv = "metrics_output.csv"
    try:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                # Convert dictionaries (for fairness metrics) to string
                # so they can be placed in a single CSV cell
                writer.writerow([key, str(value)])
        print(f"Metrics have been written to {output_csv}")
    except Exception as e:
        print(f"Failed to write metrics to CSV: {e}")


if __name__ == "__main__":
    main()
