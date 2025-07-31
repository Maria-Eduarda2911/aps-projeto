import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import xgboost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import sys

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Obesity Prediction")

# Criar dataset de demonstração se necessário
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "obesity.csv")

if not os.path.exists(DATA_PATH):
    print("Criando dataset de demonstração...")
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=1000, n_features=20, n_informative=15,
        noise=0.1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["obesity_rate"] = y
    df.to_csv(DATA_PATH, index=False)

# Carregar dados
try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar dataset: {str(e)}")
    sys.exit(1)

# Pré-processamento
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna('missing', inplace=True)

if "obesity_rate" not in df.columns:
    num_cols = df.select_dtypes(include=np.number).columns
    df.rename(columns={num_cols[0]: "obesity_rate"}, inplace=True)

# Separar dados
X = df.drop("obesity_rate", axis=1)
y = df["obesity_rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar pré-processamento
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Modelos e hiperparâmetros para otimização
models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {
            "model__fit_intercept": [True, False],
            "model__positive": [True, False],
            "model__copy_X": [True, False],
            "model__n_jobs": [-1, 1, 2, 4, 8]
        }
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "model__alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
            "model__fit_intercept": [True, False],
            "model__solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
            "model__max_iter": [100, 500, 1000, 2000, 5000]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            "model__alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
            "model__fit_intercept": [True, False],
            "model__selection": ['cyclic', 'random'],
            "model__max_iter": [100, 500, 1000, 2000, 5000]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(),
        "params": {
            "model__n_estimators": [50, 100, 200, 300, 400],
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10, 15, 20],
            "model__min_samples_leaf": [1, 2, 4, 8, 10]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(),
        "params": {
            "model__n_estimators": [50, 100, 200, 300, 400],
            "model__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
            "model__max_depth": [3, 5, 10, 20, 30],
            "model__subsample": [0.5, 0.7, 0.8, 0.9, 1.0]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(),
        "params": {
            "model__n_estimators": [50, 100, 200, 300, 400],
            "model__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
            "model__max_depth": [3, 5, 10, 20, 30],
            "model__subsample": [0.5, 0.7, 0.8, 0.9, 1.0]
        }
    }
}

# Treinar e registrar modelos
best_model = None
best_rmse = float('inf')

for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config["model"])
        ])
        
        search = RandomizedSearchCV(
            pipeline,
            config["params"],
            n_iter=50,
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        preds = best_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_pipeline, "model")
        
        print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = best_pipeline
            best_model_name = model_name

# Registrar o melhor modelo
if best_model:
    with mlflow.start_run(run_name="BEST_MODEL") as run:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("rmse", best_rmse)
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="ObesityModel")
        print(f"\nMelhor modelo: {best_model_name} com RMSE: {best_rmse:.4f}")

print("\nTreinamento concluído!")