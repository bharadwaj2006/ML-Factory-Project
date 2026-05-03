import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import shap
from imblearn.over_sampling import SMOTE
import os

def generate_synthetic_dataset(n_samples=100000):
    """Generate imbalanced synthetic dataset matching spec"""
    n_failures = int(n_samples * 0.008)  # 0.8% failure rate
    
    # Normal operations (99.2%)
    normal_data = []
    for _ in range(n_samples - n_failures):
        time = np.arange(24)
        vibration = 2.5 + 0.3 * np.sin(time/3) + np.random.normal(0, 0.3, 24)
        temperature = 65 + 5 * np.sin(time/4) + np.random.normal(0, 3, 24)
        pressure = 12.5 + np.random.normal(0, 1.5, 24)
        failure = [0] * 24
        normal_data.extend(list(zip(time, vibration, temperature, pressure, failure)))
    
    # Failure patterns (0.8%)
    failure_data = []
    for _ in range(n_failures):
        time = np.arange(24)
        # Pre-failure degradation
        vibration = 3.5 + 2 * (time > 12) + np.random.normal(0, 0.8, 24)
        temperature = 70 + 15 * (time > 15) + np.random.normal(0, 5, 24)
        pressure = 12.5 + np.abs(15 - time) * 0.3 + np.random.normal(0, 2, 24)
        failure = [(t > 20) for t in time]  # Failure at t=21
        failure_data.extend(list(zip(time, vibration, temperature, pressure, failure)))
    
    df = pd.DataFrame(normal_data + failure_data, 
                     columns=['hour', 'vibration', 'temperature', 'pressure', 'failure'])
    return df

def create_features_pipeline(df):
    """Production feature engineering pipeline"""
    features = pd.DataFrame(index=df.index)
    
    # Raw features
    features['vib'] = df['vibration']
    features['temp'] = df['temperature']
    features['press'] = df['pressure']
    
    # Rolling statistics
    for window in [3, 6, 12]:
        features[f'vib_mean_{window}h'] = df['vibration'].rolling(window, min_periods=1).mean()
        features[f'temp_mean_{window}h'] = df['temperature'].rolling(window, min_periods=1).mean()
        features[f'press_std_{window}h'] = df['pressure'].rolling(window, min_periods=1).std()
        features[f'vib_lag1'] = df['vibration'].shift(1).fillna(method='bfill')
    
    # EMA
    features['vib_ema'] = df['vibration'].ewm(span=6).mean()
    
    # Interactions
    features['vib_temp_interaction'] = features['vib'] * features['temp']
    
    return features.dropna()

# Generate data
print("Generating synthetic dataset...")
df = generate_synthetic_dataset(50000)
print(f"Dataset shape: {df.shape}, Failure rate: {df['failure'].mean():.3%}")

# Feature engineering
X = create_features_pipeline(df).fillna(0)
y = df.loc[X.index, 'failure'].values

print(f"Features shape: {X.shape}")

# Split and handle imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_train_res[y_train_res==0])/len(y_train_res[y_train_res==1]),
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train_res)

# Evaluate PR-AUC
probs = model.predict_proba(X_test_scaled)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled[:100])

# Save
os.makedirs('backend/models', exist_ok=True)
joblib.dump(model, 'backend/models/xgb_model.joblib')
joblib.dump(scaler, 'backend/models/scaler.joblib')
joblib.dump(X.columns.tolist(), 'backend/models/feature_names.joblib')

print("Model saved! PR-AUC:", pr_auc)
print("Top features:", pd.Series(X.columns)[np.argsort(model.feature_importances_)[-10:][::-1]].tolist())

if __name__ == '__main__':
    pass
