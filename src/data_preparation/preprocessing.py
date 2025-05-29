
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    df['scaled_amount'] = scaler_amount.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler_time.fit_transform(df[['Time']])
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    scaled_df = df[['scaled_time', 'scaled_amount'] + [col for col in df.columns if col not in ['scaled_time', 'scaled_amount']]]

    os.makedirs(output_path, exist_ok=True)
    scaled_df.to_csv(os.path.join(output_path, 'creditcard_cleaned.csv'), index=False)
    
    joblib.dump(scaler_amount, os.path.join(output_path, 'scaler_amount.pkl'))
    joblib.dump(scaler_time, os.path.join(output_path, 'scaler_time.pkl'))

    print("✅ Preprocessing complete. Cleaned data and scalers saved.")

# Example usage (for testing outside the notebook)
if __name__ == "__main__":
    preprocess_data("data/raw/creditcard.csv", "data/processed/")
