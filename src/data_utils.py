import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

def clean_column_names(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col.lower().replace(' ', '_')) for col in df.columns]
    return df

def basic_data_cleaning(df):
    df = clean_column_names(df)
    
    # Identificar coluna alvo
    target_candidates = ['datavalue', 'value', 'obesityrate', 'prevalence', 'rate', 'obesity_rate']
    for col in target_candidates:
        if col in df.columns:
            df.rename(columns={col: 'obesity_rate'}, inplace=True)
            break
    else:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            df.rename(columns={num_cols[0]: 'obesity_rate'}, inplace=True)
    
    df.dropna(axis=1, how='all', inplace=True)
    return df.drop_duplicates()

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns=None, categorical_threshold=15):
        self.date_columns = date_columns or []
        self.categorical_threshold = categorical_threshold
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        # Tratamento de datas
        for col in self.date_columns:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors='coerce')
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
        
        # Conversão de tipos
        for col in X.select_dtypes(include='object').columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='ignore')
            except:
                pass
            
            unique_count = X[col].nunique()
            if 1 < unique_count < self.categorical_threshold:
                X[col] = X[col].astype('category')
        
        return X

def feature_engineering(df):
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Criação de features
    if 'yearstart' in df.columns and 'yearend' in df.columns:
        df['year_avg'] = (df['yearstart'].astype(float) + df['yearend'].astype(float)) / 2
        
    # Mapeamento de regiões
    if 'locationdesc' in df.columns:
        regions = {
            'northeast': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire',
                          'Rhode Island', 'Vermont', 'New Jersey', 'New York', 'Pennsylvania'],
            'midwest': ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin',
                        'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
            'south': ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina',
                      'South Carolina', 'Virginia', 'District of Columbia', 'West Virginia',
                      'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas'],
            'west': ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming',
                     'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
        }
        
        def map_region(location):
            for region, states in regions.items():
                if location in states:
                    return region
            return 'other'
            
        df['region'] = df['locationdesc'].apply(map_region)
    
    return df