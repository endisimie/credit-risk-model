import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

class TransactionFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

        # Extract datetime features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        # Aggregate features per CustomerId
        agg_df = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        })
        agg_df.columns = ['TotalAmount', 'AverageAmount', 'TransactionCount', 'AmountStdDev']
        agg_df.reset_index(inplace=True)

        # Merge aggregate features back
        df = df.merge(agg_df, on='CustomerId', how='left')

        return df

# List of categorical columns to encode
categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']

# Define preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('feature_gen', TransactionFeatureGenerator()),
    ('column_transform', ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('impute_scale', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), ['Amount', 'TotalAmount', 'AverageAmount', 'TransactionCount', 'AmountStdDev',
             'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'])
    ], remainder='drop'))
])




# Example usage:
#df_raw = pd.read_csv("path_to_raw_data.csv")
#df_transformed = preprocessing_pipeline.fit_transform(df_raw)

# If you want to persist this for modeling:
# from joblib import dump
# dump(preprocessing_pipeline, 'src/pipeline/feature_pipeline.joblib')
