import gspread
import requests
import csv
import io
from google.auth import default
from google.colab import auth

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
import numpy as np
from fairlearn.metrics import *

creds, _ = default()
gc = gspread.authorize(creds)

sheet_url = "https://docs.google.com/spreadsheets/d/16uBL9zzLeM-jvESowqEoRKxSkrUMVVNuP8ATXyIJBw0/edit?gid=1714650698#gid=1714650698"
sheet = gc.open_by_url(sheet_url)
worksheet = sheet.get_worksheet(0)

# we get these from the google sheets
diagnosis_correct = worksheet.col_values(4)[1:]
diagnosis_informative = worksheet.col_values(5)[1:]
diagnosis_helpful = worksheet.col_values(6)[1:]
diagnosis_understand = worksheet.col_values(7)[1:]


total_images = 100  ###CHANGE TO 298 FOR FULL DATASET

y_true_correct = np.ones(total_images, dtype=int)
y_pred_correct = np.zeros(total_images, dtype=int)


y_true_informative = np.ones(total_images, dtype=int)
y_pred_informative = np.zeros(total_images, dtype=int)

y_true_helpful = np.ones(total_images, dtype=int)
y_pred_helpful = np.zeros(total_images, dtype=int)

y_true_understand = np.ones(total_images, dtype=int)
y_pred_understand = np.zeros(total_images, dtype=int)

skin_tone = np.zeros(total_images, dtype=int)

###adding all values to the four categories
# correct
for i, value in enumerate(diagnosis_correct):
    if i < total_images and value.lower() == "yes":
        y_pred_correct[i] = 1

# inform
for i, value in enumerate(diagnosis_informative):
    if i < total_images and value.lower() == "yes":
        y_pred_informative[i] = 1

# helpful
for i, value in enumerate(diagnosis_helpful):
    if i < total_images and value.lower() == "yes":
        y_pred_helpful[i] = 1

# understand
for i, value in enumerate(diagnosis_understand):
    if i < total_images and value.lower() == "yes":
        y_pred_understand[i] = 1

ninth_column_values = worksheet.col_values(8)[1:][:100]

csv_url = "https://raw.githubusercontent.com/buiswrld/SkinGPT-4/refs/heads/main/data/10-sample.csv"
response = requests.get(csv_url)
response.raise_for_status()  # Raise an exception for bad responses
csv_data = response.content.decode("utf-8")
reader = csv.reader(io.StringIO(csv_data))
next(reader)  # Skip the header row of the CSV
count = 0

for index, google_sheet_value in enumerate(ninth_column_values):
    if index >= 100:
        break
    for idx, line in enumerate(csv_data.splitlines()):
        row = line.split(",")
        if google_sheet_value in row[0]:
            try:
                skin_tone[index] = int(row[2])
                break
            except (IndexError, ValueError):
                count = count + 1
                break

################CALCULATING HERE################

# Correct
equalized_odds_contact_derm_correct = equalized_odds_difference(
    y_true_correct[:50], y_pred_correct[:50], sensitive_features=skin_tone[:50]
)
demographic_parity_contact_derm_correct = demographic_parity_difference(
    y_true_correct[:50],
    y_pred_correct[:50],
    sensitive_features=skin_tone[:50],
    method="to_overall",
)

print(f"equalized odds for contact dermatitis: {equalized_odds_contact_derm_correct}")
print(
    f"demographic parity for contact dermatitis: {demographic_parity_contact_derm_correct}"
)

equalized_odds_eczema_correct = equalized_odds_difference(
    y_true_correct[50:100], y_pred_correct[50:100], sensitive_features=skin_tone[50:100]
)
demographic_parity_eczema_correct = demographic_parity_difference(
    y_true_correct[50:100],
    y_pred_correct[50:100],
    sensitive_features=skin_tone[50:100],
    method="to_overall",
)

print(f"equalized odds for eczema: {equalized_odds_eczema_correct}")
print(f"demographic parity for eczema: {demographic_parity_eczema_correct}")


# Informative
equalized_odds_contact_derm_informative = equalized_odds_difference(
    y_true_informative[:50], y_pred_informative[:50], sensitive_features=skin_tone[:50]
)
demographic_parity_contact_derm_informative = demographic_parity_difference(
    y_true_informative[:50],
    y_pred_informative[:50],
    sensitive_features=skin_tone[:50],
    method="to_overall",
)

print(
    f"equalized odds for contact dermatitis (informative): {equalized_odds_contact_derm_informative}"
)
print(
    f"demographic parity for contact dermatitis (informative): {demographic_parity_contact_derm_informative}"
)

equalized_odds_eczema_informative = equalized_odds_difference(
    y_true_informative[50:100],
    y_pred_informative[50:100],
    sensitive_features=skin_tone[50:100],
)
demographic_parity_eczema_informative = demographic_parity_difference(
    y_true_informative[50:100],
    y_pred_informative[50:100],
    sensitive_features=skin_tone[50:100],
    method="to_overall",
)

print(f"equalized odds for eczema (informative): {equalized_odds_eczema_informative}")
print(
    f"demographic parity for eczema (informative): {demographic_parity_eczema_informative}"
)

# Helpful
equalized_odds_contact_derm_helpful = equalized_odds_difference(
    y_true_helpful[:50], y_pred_helpful[:50], sensitive_features=skin_tone[:50]
)
demographic_parity_contact_derm_helpful = demographic_parity_difference(
    y_true_helpful[:50],
    y_pred_helpful[:50],
    sensitive_features=skin_tone[:50],
    method="to_overall",
)

print(
    f"equalized odds for contact dermatitis (helpful): {equalized_odds_contact_derm_helpful}"
)
print(
    f"demographic parity for contact dermatitis (helpful): {demographic_parity_contact_derm_helpful}"
)

equalized_odds_eczema_helpful = equalized_odds_difference(
    y_true_helpful[50:100], y_pred_helpful[50:100], sensitive_features=skin_tone[50:100]
)
demographic_parity_eczema_helpful = demographic_parity_difference(
    y_true_helpful[50:100],
    y_pred_helpful[50:100],
    sensitive_features=skin_tone[50:100],
    method="to_overall",
)

print(f"equalized odds for eczema (helpful): {equalized_odds_eczema_helpful}")
print(f"demographic parity for eczema (helpful): {demographic_parity_eczema_helpful}")

# Understand
equalized_odds_contact_derm_understand = equalized_odds_difference(
    y_true_understand[:50], y_pred_understand[:50], sensitive_features=skin_tone[:50]
)
demographic_parity_contact_derm_understand = demographic_parity_difference(
    y_true_understand[:50],
    y_pred_understand[:50],
    sensitive_features=skin_tone[:50],
    method="to_overall",
)

print(
    f"equalized odds for contact dermatitis (understand): {equalized_odds_contact_derm_understand}"
)
print(
    f"demographic parity for contact dermatitis (understand): {demographic_parity_contact_derm_understand}"
)

equalized_odds_eczema_understand = equalized_odds_difference(
    y_true_understand[50:100],
    y_pred_understand[50:100],
    sensitive_features=skin_tone[50:100],
)
demographic_parity_eczema_understand = demographic_parity_difference(
    y_true_understand[50:100],
    y_pred_understand[50:100],
    sensitive_features=skin_tone[50:100],
    method="to_overall",
)

print(f"equalized odds for eczema (understand): {equalized_odds_eczema_understand}")
print(
    f"demographic parity for eczema (understand): {demographic_parity_eczema_understand}"
)
