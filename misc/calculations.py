import gspread
import requests
import csv
import io
import logging
import sys
import numpy as np
from google.auth import default
from google.colab import auth
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def authenticate_google():
    """
    Authenticates the user with Google credentials.

    Returns:
        gspread.Client: Authorized gspread client.
    """
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
    logging.info("Authenticated and authorized with Google.")
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
        logging.info(f"Fetched worksheet {worksheet_index} from {sheet_url}")
        return worksheet
    except Exception as e:
        logging.error(f"Failed to fetch Google Sheet: {e}")
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
        "y_true_correct": np.ones(total_images, dtype=int),
        "y_pred_correct": np.zeros(total_images, dtype=int),
        "y_true_informative": np.ones(total_images, dtype=int),
        "y_pred_informative": np.zeros(total_images, dtype=int),
        "y_true_helpful": np.ones(total_images, dtype=int),
        "y_pred_helpful": np.zeros(total_images, dtype=int),
        "y_true_understand": np.ones(total_images, dtype=int),
        "y_pred_understand": np.zeros(total_images, dtype=int),
        "skin_tone": np.zeros(total_images, dtype=int),
    }
    logging.info("Initialized prediction and true label arrays.")
    return arrays


def update_predictions(conditions, y_pred, total_images):
    """
    Updates prediction arrays based on conditions.

    Args:
        conditions (list): List of condition values from Google Sheets.
        y_pred (np.ndarray): Prediction array to update.
        total_images (int): Total number of images.
    """
    y_pred[: len(conditions[:total_images])] = np.array(
        [1 if value.lower() == "yes" else 0 for value in conditions[:total_images]]
    )
    logging.debug(f"Updated predictions for {len(conditions[:total_images])} images.")


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
        header = next(reader)  # Skip header
        data = list(reader)
        logging.info(f"Fetched and parsed CSV data from {csv_url}")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch CSV data: {e}")
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
        skin_tone_array[i] = lookup.get(value, 0)
    logging.info("Mapped skin tone values.")


def calculate_fairness_metrics(y_true, y_pred, sensitive_features, description):
    """
    Calculates and prints fairness metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        sensitive_features (np.ndarray): Sensitive features array.
        description (str): Description for logging.
    """
    try:
        eod = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        dpd = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features, method="to_overall"
        )
        logging.info(f"Equalized Odds ({description}): {eod}")
        logging.info(f"Demographic Parity ({description}): {dpd}")
    except Exception as e:
        logging.error(f"Failed to calculate fairness metrics for {description}: {e}")


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
        rate = counter / total_entries
        logging.info(f"Hallucinations rate: {rate}")
        return rate
    except Exception as e:
        logging.error(f"Failed to calculate hallucination rate: {e}")
        return 0.0


def main():
    # Configuration
    sheet_url = "https://docs.google.com/spreadsheets/d/16uBL9zzLeM-jvESowqEoRKxSkrUMVVNuP8ATXyIJBw0/edit#gid=1714650698"
    csv_url = (
        "https://raw.githubusercontent.com/buiswrld/SkinGPT-4/main/data/10-sample.csv"
    )
    total_images = 100  # Change to 298 for full dataset
    split_index = 50

    # Authenticate and fetch Google Sheet
    gc = authenticate_google()
    worksheet = fetch_google_sheet(gc, sheet_url)

    # Initialize prediction and true label arrays
    arrays = initialize_arrays(total_images)

    # Fetch condition columns from Google Sheets
    diagnosis_correct = worksheet.col_values(4)[1:]
    diagnosis_informative = worksheet.col_values(5)[1:]
    diagnosis_helpful = worksheet.col_values(6)[1:]
    diagnosis_understand = worksheet.col_values(7)[1:]

    # Update prediction arrays using vectorized operations
    update_predictions(diagnosis_correct, arrays["y_pred_correct"], total_images)
    update_predictions(
        diagnosis_informative, arrays["y_pred_informative"], total_images
    )
    update_predictions(diagnosis_helpful, arrays["y_pred_helpful"], total_images)
    update_predictions(diagnosis_understand, arrays["y_pred_understand"], total_images)

    # Fetch and map skin tone data
    csv_data = fetch_csv_data(csv_url)
    ninth_column_values = worksheet.col_values(8)[1:]
    map_skin_tone(ninth_column_values, csv_data, arrays["skin_tone"], total_images)

    # Calculate fairness metrics for different conditions
    calculate_fairness_metrics(
        arrays["y_true_correct"][:split_index],
        arrays["y_pred_correct"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Correct)",
    )

    calculate_fairness_metrics(
        arrays["y_true_correct"][split_index:total_images],
        arrays["y_pred_correct"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Correct)",
    )

    calculate_fairness_metrics(
        arrays["y_true_informative"][:split_index],
        arrays["y_pred_informative"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Informative)",
    )

    calculate_fairness_metrics(
        arrays["y_true_informative"][split_index:total_images],
        arrays["y_pred_informative"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Informative)",
    )

    calculate_fairness_metrics(
        arrays["y_true_helpful"][:split_index],
        arrays["y_pred_helpful"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Helpful)",
    )

    calculate_fairness_metrics(
        arrays["y_true_helpful"][split_index:total_images],
        arrays["y_pred_helpful"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Helpful)",
    )

    calculate_fairness_metrics(
        arrays["y_true_understand"][:split_index],
        arrays["y_pred_understand"][:split_index],
        arrays["skin_tone"][:split_index],
        "Contact Dermatitis (Understand)",
    )

    calculate_fairness_metrics(
        arrays["y_true_understand"][split_index:total_images],
        arrays["y_pred_understand"][split_index:total_images],
        arrays["skin_tone"][split_index:total_images],
        "Eczema (Understand)",
    )

    # Calculate hallucination rate
    hallucination_rate = calculate_hallucination_rate(worksheet, total_entries=298)
    logging.info(f"Final Hallucinations Rate: {hallucination_rate}")


if __name__ == "__main__":
    main()
