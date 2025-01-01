import gspread
import requests
import csv
import io
from google.auth import default
!pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
from google.colab import auth
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
import numpy as np
!pip install fairlearn

import numpy as np
from fairlearn.metrics import MetricFrame, equalized_odds_difference, demographic_parity_difference# Step 1: Define the conditions and dataset ranges
import gspread
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

sheet_url = "https://docs.google.com/spreadsheets/d/1O1jKNm-_KRsafYoSxS_sd-8rTaYlIce1FyYpcae_15g/edit?gid=1714650698#gid=1714650698"
sheet = gc.open_by_url(sheet_url)
worksheet = sheet.get_worksheet(0)
diagnosis_correct = worksheet.col_values(4)[1:]  #skip header row
diagnosis_informative = worksheet.col_values(5)[1:]
diagnosis_helpful = worksheet.col_values(6)[1:]
diagnosis_understand = worksheet.col_values(4)[1:]

]# Dataset ranges for each condition
ranges = {
    "Allergic Contact Dermatitis": (0, 50),
    "Eczema": (50, 100),
    #"Impetigo": (100, 149),
    #"Psoriasis": (149, 199),
    #"Tinea": (199, 248),
    #"Urticaria": (248, 298),

}# Step 2: Simulate y_true, y_pred, and skin tone labels for the dataset
total_images = 100 ###CHANGE TO 298 FOR FULL DATASET
y_true_correct = np.zeros(total_images)  # Initialize y_true for all conditions
y_pred_correct = np.zeros(total_images)  # Initialize y_pred for all conditionsfor condition, (start, end) in ranges.items():

y_true_informative = np.zeros(total_images)
y_pred_informative = np.zeros(total_images)

y_true_helpful = np.zeros(total_images)
y_pred_helpful = np.zeros(total_images)

y_true_understand = np.zeros(total_images)
y_pred_understand = np.zeros(total_images)

#y_true_hallucinations = np.zeros(total_images)
#y_pred_hallucinations = np.zeros(total_images)

skin_tone = np.zeros(total_images)

###adding all values to the four question categories: correct, inform, helpful, and understand
#correct
for i, value in enumerate(diagnosis_correct):
  if i < len(y_pred_correct): #check if i is in the valid index range before proceeding
    if value.lower() == "yes":
      y_pred_correct[i] = 1 #if yes 1, if no 0
    else:
      y_pred_correct[i] = 0
    y_true_correct[i] = 1 #correct always set to 1
  else:
    break
#inform
for i, value in enumerate(diagnosis_informative):
  if i < len(y_pred_correct): #check if i is in the valid index range before proceeding
    if value.lower() == "yes":
      y_pred_informative[i] = 1
    else:
      y_pred_informative[i] = 0
    y_true_informative[i] = 1
  else:
    break
#helpful
for i, value in enumerate(diagnosis_helpful):
  if i < len(y_pred_correct): #check if i is in the valid index range before proceeding
    if value.lower() == "yes":
      y_pred_helpful[i] = 1
    else:
      y_pred_helpful[i] = 0
    y_true_helpful[i] = 1
  else:
    break
#understand
for i, value in enumerate(diagnosis_understand):
  if i < len(y_pred_correct): #check if i is in the valid index range before proceeding
    if value.lower() == "yes":
      y_pred_understand[i] = 1
    else:
      y_pred_understand[i] = 0
    y_true_understand[i] = 1
  else:
    break

#creating skin_tone numbers
# Get values from the 9th column of the Google Sheet
ninth_column_values = worksheet.col_values(9)[1:]

skin_tone = np.zeros(100) ###CHANGE TO 298 FOR FULL DATASET

#creating skin_tone numbers
csv_url = "https://raw.githubusercontent.com/buiswrld/SkinGPT-4/refs/heads/main/data/10-sample.csv"
response = requests.get(csv_url)
response.raise_for_status()  # Raise an exception for bad responses
csv_data = response.content.decode('utf-8')
reader = csv.reader(io.StringIO(csv_data))
next(reader)  # Skip the header row of the CSV
count = 0
bruh = 0
for index, google_sheet_value in enumerate(ninth_column_values):
    bruh = bruh + 1
    for line in csv_data.splitlines():  # Iterate over lines in the CSV data
        row = line.split(',')  # Split the line by commas
        if google_sheet_value in row[0]:  # Check the first element (image path)
            try:
                skin_tone[index] = int(row[2])  # Get value from the third element
                break  # Break inner loop once found
            except (IndexError, ValueError): #means it tried accessing smth that wasnt in github, not needed when using full dataset
                #print(f"Warning: Issue with CSV row for {google_sheet_value}: {line}")
                count = count + 1
                break
print(count)

################CALCULATING HERE################
###Psoriasis
equalized_odds = equalized_odds_difference(y_true_correct[:100], y_pred_correct[:100], sensitive_features=skin_tone[:100])
demographic_parity = demographic_parity_difference(y_true_correct[:100], y_pred_correct[:100], sensitive_features=skin_tone[:100])    
print(f"Equalized Odds Correct Difference: {equalized_odds:.4f}")
print(f"Demographic Parity Correct Difference: {demographic_parity:.4f}")

equalized_odds = equalized_odds_difference(y_true_informative[:100], y_pred_informative[:100], sensitive_features=skin_tone[:100])
demographic_parity = demographic_parity_difference(y_true_informative[:100], y_pred_informative[:100], sensitive_features=skin_tone[:100])    
print(f"Equalized Odds Informative Difference: {equalized_odds:.4f}")
print(f"Demographic Parity Informative Difference: {demographic_parity:.4f}")

equalized_odds = equalized_odds_difference(y_true_helpful[:100], y_pred_helpful[:100], sensitive_features=skin_tone[:100])
demographic_parity = demographic_parity_difference(y_true_helpful[:100], y_pred_helpful[:100], sensitive_features=skin_tone[:100])    
print(f"Equalized Odds Helpful Difference: {equalized_odds:.4f}")
print(f"Demographic Parity Helpful Difference: {demographic_parity:.4f}")

equalized_odds = equalized_odds_difference(y_true_understand[:100], y_pred_understand[:100], sensitive_features=skin_tone[:100])
demographic_parity = demographic_parity_difference(y_true_understand[:100], y_pred_understand[:100], sensitive_features=skin_tone[:100])    
print(f"Equalized Odds Understand Difference: {equalized_odds:.4f}")
print(f"Demographic Parity Understand Difference: {demographic_parity:.4f}")

