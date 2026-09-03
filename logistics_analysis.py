# Logistics Data Analyst Internship
# Week 1: Strategic Planning and Data Exploration in Logistics
# Project: Delivery Performance Analysis and Route Optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load logistics dataset
df = pd.read_csv("logistics_data.csv")

# Display first few records
print(df.head())

# Check dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Fill missing distance values with the median
if "Distance_km" in df.columns:
    df["Distance_km"] = df["Distance_km"].fillna(
        df["Distance_km"].median()
    )

# Calculate key performance indicators (KPIs)

# On-Time Delivery Rate
if "Delivery_Status" in df.columns:
    on_time_rate = (
        (df["Delivery_Status"] == "On Time").mean() * 100
    )
    print("On-Time Delivery Rate:", round(on_time_rate, 2), "%")

# Average Delivery Time
if "Delivery_Time_Hours" in df.columns:
    average_delivery_time = df["Delivery_Time_Hours"].mean()
    print(
        "Average Delivery Time:",
        round(average_delivery_time, 2),
        "hours"
    )

# Average Transportation Cost
if "Transportation_Cost" in df.columns:
    average_cost = df["Transportation_Cost"].mean()
    print(
        "Average Transportation Cost:",
        round(average_cost, 2)
    )

# Exploratory Data Analysis
print(df.describe())

# Visualize delivery time distribution
if "Delivery_Time_Hours" in df.columns:
    plt.hist(df["Delivery_Time_Hours"])
    plt.xlabel("Delivery Time (Hours)")
    plt.ylabel("Number of Deliveries")
    plt.title("Distribution of Delivery Times")
    plt.show()

# Simple regression model
# Predict delivery time using distance
if "Distance_km" in df.columns and "Delivery_Time_Hours" in df.columns:

    X = df[["Distance_km"]]
    y = df["Delivery_Time_Hours"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Predicted Delivery Times:")
    print(predictions[:5])

# Route Optimization Concept
# The final route optimization stage would consider:
# 1. Distance between delivery locations
# 2. Vehicle capacity
# 3. Delivery time windows
# 4. Transportation cost
# 5. Traffic and route conditions

print("Logistics analysis workflow completed.")
