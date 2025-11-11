import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set nama eksperimen
mlflow.set_experiment("CI_Model_Weather")

# Load dataset hasil preprocessing
dataset_path = "weather_preprocessed.csv" 
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"File {dataset_path} tidak ditemukan!")

data = pd.read_csv(dataset_path)

target_column = "Temperature (C)" 

if target_column not in data.columns:
     raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")

X = data.drop(columns=[target_column])
y = data[target_column]
X = X.select_dtypes(include=["number"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
input_example = X_train.head(5)

# Training model + MLflow autolog
mlflow.sklearn.autolog()

print("🚀 Melatih model RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MSE: {mse:.4f}")
print(f"📈 R2 Score: {r2:.4f}")

# Log metrik secara manual 
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)
mlflow.sklearn.log_model(model, "model", input_example=input_example)

# Ambil ID run yang AKTIF 
run = mlflow.active_run()
if run:
    run_id = run.info.run_id
    print(f"✅ Pelatihan selesai. Run ID: {run_id}")
    
    # Simpan run_id ke file agar bisa dibaca GitHub Actions
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print("📂 run_id.txt berhasil disimpan.")
else:
    print("⚠️ Peringatan: Tidak dapat menemukan run MLflow yang aktif.")
    raise RuntimeError("Gagal mendapatkan run MLflow yang aktif.")