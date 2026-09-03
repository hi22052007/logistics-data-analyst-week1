# Logistics Data Analyst Internship
# Week 2: Data Collection, Cleaning and Preprocessing
# Project: Last-Mile Delivery Data

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. DATA COLLECTION / LOADING
# --------------------------------------------------

# Load the logistics dataset
df = pd.read_csv("logistics_data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())


# --------------------------------------------------
# 2. INITIAL DATA INSPECTION
# --------------------------------------------------

print("\nData Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nSummary Statistics:")
print(df.describe(include="all"))


# --------------------------------------------------
# 3. REMOVE DUPLICATE RECORDS
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicate rows removed:", before - after)


# --------------------------------------------------
# 4. CORRECT DATA TYPES
# --------------------------------------------------

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

numeric_columns = [
    "Distance_km",
    "Delivery_Time_Hours",
    "Transportation_Cost"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# --------------------------------------------------
# 5. HANDLE MISSING VALUES
# --------------------------------------------------

if "Distance_km" in df.columns:
    df["Distance_km"] = df["Distance_km"].fillna(
        df["Distance_km"].median()
    )

if "Vehicle_Type" in df.columns:
    df["Vehicle_Type"] = df["Vehicle_Type"].fillna(
        "Unknown"
    )


# --------------------------------------------------
# 6. STANDARDIZE CATEGORICAL VALUES
# --------------------------------------------------

if "Vehicle_Type" in df.columns:
    df["Vehicle_Type"] = (
        df["Vehicle_Type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# --------------------------------------------------
# 7. OUTLIER DETECTION USING IQR
# --------------------------------------------------

def find_iqr_outliers(data, column):

    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    return data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]


if "Delivery_Time_Hours" in df.columns:

    outliers = find_iqr_outliers(
        df,
        "Delivery_Time_Hours"
    )

    print(
        "\nPotential delivery-time outliers:",
        len(outliers)
    )


# --------------------------------------------------
# 8. MIN-MAX NORMALIZATION
# --------------------------------------------------

features = [
    "Distance_km",
    "Delivery_Time_Hours",
    "Transportation_Cost"
]

available_features = [
    column
    for column in features
    if column in df.columns
]

if available_features:

    scaler = MinMaxScaler()

    df[available_features] = scaler.fit_transform(
        df[available_features]
    )

    print("\nMin-Max normalization completed.")


# --------------------------------------------------
# 9. FINAL DATA QUALITY CHECK
# --------------------------------------------------

print("\nRemaining Duplicate Rows:")
print(df.duplicated().sum())

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nFinal Data Types:")
print(df.dtypes)


# --------------------------------------------------
# 10. SAVE CLEANED DATA
# --------------------------------------------------

df.to_csv(
    "logistics_data_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")


# --------------------------------------------------
# END OF WEEK 2 PREPROCESSING PIPELINE
# --------------------------------------------------

print("\nWeek 2 data preprocessing workflow completed.")
