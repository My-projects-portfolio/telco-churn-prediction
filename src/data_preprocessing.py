import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.dropna()  # Drop rows with missing values

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop customerID
    df = df.drop('customerID', axis=1)

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df
