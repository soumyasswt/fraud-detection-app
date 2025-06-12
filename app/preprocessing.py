import pandas as pd

schema_map = {
    'amount': ['amount', 'amt', 'transaction_amount'],
    'timestamp': ['time', 'timestamp', 'datetime'],
    'user': ['user_id', 'customer_id', 'userid'],
    'location': ['location', 'city', 'region'],
    'device': ['device', 'device_type'],
    'fraud': ['is_fraud', 'fraud', 'fraud_flag', 'class']
}

def auto_map_columns(df):
    mapped = {}
    for key, variants in schema_map.items():
        for col in df.columns:
            if col.lower() in variants:
                mapped[key] = col
                break
    return mapped

def preprocess_df(df):
    df = df.copy()

    # Clean 'fraud' column if it exists
    if 'fraud' in df.columns:
        df['fraud'] = df['fraud'].replace(['Unknown', 'N/A', 'NA', 'null', None], -1)
        df = df[df['fraud'].astype(str).str.isnumeric()]
        df['fraud'] = df['fraud'].astype(int)
    
    # Optional: Fill other missing values
    df = df.fillna("unknown")

    # Convert categorical columns to string
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    return df

