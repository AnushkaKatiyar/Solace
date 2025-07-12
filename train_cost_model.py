import pandas as pd
import numpy as np
import re
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from xgboost import XGBRegressor
import nltk
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# --- Configuration ---
DATA_PATH = "data/Capital_Project_Schedules_and_Budgets.csv"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(DATA_PATH)

# --- Feature Engineering ---
df["start_date"] = pd.to_datetime(df["Project Phase Actual Start Date"], errors="coerce", format="%m/%d/%Y")
actual_end = pd.to_datetime(df["Project Phase Actual End Date"], errors="coerce", format="%m/%d/%Y")
planned_end = pd.to_datetime(df["Project Phase Planned End Date"], errors="coerce", format="%m/%d/%Y")
df["end_date"] = actual_end.combine_first(planned_end)
df["project_status"] = df["Project Status Name"].str.strip().str.upper()
df.loc[df["project_status"] == "PNS", ["start_date", "end_date"]] = pd.NaT
df["end_date_missing"] = df["end_date"].isna()
df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
df["timeline_status"] = np.where(
    df["project_status"] == "PNS", "Not Started",
    np.where(df["end_date"].isna(), "Incomplete", "Available")
)

df["actual_spend"] = pd.to_numeric(df["Total Phase Actual Spending Amount"], errors="coerce")
df["estimated_spend"] = pd.to_numeric(df["Final Estimate of Actual Costs Through End of Phase Amount"], errors="coerce")
df["budgeted_spend"] = pd.to_numeric(df["Project Budget Amount"], errors="coerce")
df["final_cost"] = df["actual_spend"].combine_first(df["estimated_spend"]).combine_first(df["budgeted_spend"]).fillna(0)

df['fiscal_year_short'] = df['Project Description'].str.extract(r'FY(\d{2})', expand=False)
df['fiscal_year_num'] = df['fiscal_year_short'].astype(float) + 2000
inflation_map = {2020: 1.18, 2021: 1.14, 2022: 1.10, 2023: 1.06, 2024: 1.00, 2025: 0.97}
df['inflation_factor'] = df['fiscal_year_num'].map(inflation_map)
df['adjusted_cost'] = df['final_cost'] * df['inflation_factor']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['description_clean'] = df['Project Description'].apply(clean_text)
df['description_no_stopwords'] = df['description_clean'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

df['cost_to_predict'] = df['adjusted_cost'].combine_first(df['final_cost'])
df['description_no_stopwords'] = df['description_no_stopwords'].fillna('')
df['Project Phase Name'] = df['Project Phase Name'].fillna('Unknown')

df_model = df[['cost_to_predict', 'duration_days', 'description_no_stopwords', 
               'Project Phase Name', 'project_status', 'timeline_status', 'end_date_missing']].dropna(subset=['cost_to_predict'])

# --- Embeddings & Encoders ---
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_embeddings(text_series):
    return bert_model.encode(text_series.tolist(), show_progress_bar=True)

cat_cols = ['Project Phase Name', 'project_status', 'timeline_status', 'end_date_missing']
num_cols = ['duration_days']

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

ohe.fit(df_model[cat_cols])
scaler.fit(df_model[num_cols])

def prepare_features(df_sub):
    bert_embeddings = get_bert_embeddings(df_sub['description_no_stopwords'])
    cat_feats = ohe.transform(df_sub[cat_cols])
    num_feats = scaler.transform(df_sub[num_cols])
    return np.hstack([bert_embeddings, cat_feats, num_feats])

df_model['cost_bucket'] = pd.qcut(df_model['cost_to_predict'], q=4, duplicates='drop')

def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'verbosity': 0
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    return rmse

# --- Train Models by Bucket ---
results = {}
bucket_map = {0: 'low_custom.pkl', 1: 'mid_custom.pkl', 2: 'high_custom.pkl'}

for idx, bucket in enumerate(df_model['cost_bucket'].cat.categories):
    df_sub = df_model[df_model['cost_bucket'] == bucket]
    if len(df_sub) < 50:
        print(f"Skipping bucket {bucket} (only {len(df_sub)} rows)")
        continue

    print(f"\n📦 Training model for cost bucket: {bucket}")
    X_sub = prepare_features(df_sub)
    y_sub = df_sub['cost_to_predict'].values
    X_train, _, y_train, _ = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)

    best_params = study.best_params
    best_params.update({'random_state': 42, 'objective': 'reg:squarederror', 'verbosity': 0})
    model = XGBRegressor(**best_params)
    model.fit(X_sub, y_sub)

    with open(os.path.join(MODEL_DIR, bucket_map[idx]), 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Saved: {bucket_map[idx]}")

# --- Save Preprocessors ---
with open(os.path.join(MODEL_DIR, 'ohe.pkl'), 'wb') as f:
    pickle.dump(ohe, f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("✅ All models and preprocessors saved in 'models/' folder.")
