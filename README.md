# Advanced Fraud Detection Analysis

This repository contains a comprehensive analysis of fraud detection using a complex synthetic dataset. The analysis includes advanced feature engineering, machine learning modeling, anomaly detection, and various visualizations.

## Overview

The goal of this project is to build a robust fraud detection system using advanced machine learning techniques. The dataset consists of 10,000 transactions with various features such as transaction amount, date, type, merchant, customer ID, location, time of day, device type, and transaction status. The project includes:

1. **Feature Engineering**: Creating sophisticated features like rolling averages, transaction velocity, etc.
2. **Correlation Analysis**: Identifying correlations between features and fraud.
3. **Machine Learning Models**: Training and evaluating models such as Gradient Boosting Classifier.
4. **Anomaly Detection**: Using Isolation Forest for detecting anomalies.
5. **Time Series Analysis**: Analyzing trends in fraud over time.
6. **Visualization**: Generating detailed visualizations to gain insights.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/advanced-fraud-detection.git
    cd advanced-fraud-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset is synthetically generated and saved as `complex_fraud_detection_data.xlsx`. It contains the following features:

- TransactionID: Unique ID for the transaction
- Amount: Transaction amount
- TransactionDate: Date and time of the transaction
- TransactionType: Type of transaction (e.g., Purchase, Withdrawal, Transfer)
- IsFraud: Indicator of whether the transaction was fraudulent (1 for fraud, 0 for non-fraud)
- Merchant: Merchant where the transaction occurred
- CustomerID: Unique ID for the customer
- Location: Location of the transaction
- TimeOfDay: Time of the day the transaction occurred (Morning, Afternoon, Evening, Night)
- DeviceType: Type of device used for the transaction (Mobile, Desktop, Tablet)
- TransactionStatus: Status of the transaction (Completed, Pending, Failed)
- Additional features like TransactionHour, TransactionDay, RollingAmount_1h, RollingAmount_24h, and TransactionVelocity

## Usage

The main analysis is contained in the Jupyter notebook or Python script provided. You can run the analysis by executing the script or running the cells in the notebook.

## Results


![1](https://github.com/sankalpsaoji98/Fraud_Detection/assets/26198596/6f59649a-6c88-4c38-a66c-3fdf043f6d1a)
![2](https://github.com/sankalpsaoji98/Fraud_Detection/assets/26198596/9cd3dd79-9b2a-4348-8090-ce8dc16df4f1)
![3](https://github.com/sankalpsaoji98/Fraud_Detection/assets/26198596/394f9b0a-283a-47b9-aed3-eede432e8780)
![5](https://github.com/sankalpsaoji98/Fraud_Detection/assets/26198596/805a380f-7c24-4a70-9fdf-1820b04427ea)


### Example

```python
# Import necessary libraries and load the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_excel('complex_fraud_detection_data.xlsx')

# Your analysis code here...

## Example


