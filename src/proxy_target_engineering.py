import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from joblib import dump

# Load the raw data
df = pd.read_csv("../data/raw/data.csv")
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

# Step 1: Calculate RFM Features
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).reset_index()
rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

# Step 2: Scale RFM Features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 3: K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Step 4: Identify High-Risk Cluster
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).sort_values(by=['Frequency', 'Monetary'], ascending=[True, True])
high_risk_cluster = cluster_summary.index[0]
rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

# Step 5: Merge Target into Main Dataset
df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Step 6: Save updated data
df.to_csv("../data/processed/data_with_high_risk_label.csv", index=False)

# Step 7: Preprocessing and Save Processed Splits
from feature_engineering_pipeline import preprocessing_pipeline

y = df['is_high_risk']
X = df.drop(columns=['is_high_risk'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

dump(preprocessing_pipeline, '../models/feature_pipeline.joblib')
pd.DataFrame(X_train_processed).to_csv('../data/processed/models/X_train_processed.csv', index=False)
pd.DataFrame(X_test_processed).to_csv('../data/processed/models/X_test_processed.csv', index=False)
pd.Series(y_train).to_csv('../data/processed/models/y_train.csv', index=False)
pd.Series(y_test).to_csv('../data/processed/models/y_test.csv', index=False)

print("✅ Task 4 complete: RFM calculated, high-risk label assigned, and data saved.")
