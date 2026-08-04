import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Resolve model path safely for local & Render deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/laptop_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "../models/model.pkl")

model = joblib.load(MODEL_PATH)

# Load options for dropdowns dynamically from dataset
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/laptops_clean.csv")
if os.path.exists(DATA_PATH):
    raw_df = pd.read_csv(DATA_PATH)
    BRANDS = sorted(raw_df["Brand"].dropna().unique().tolist())
    STATUSES = sorted(raw_df["Status"].dropna().unique().tolist())
    STORAGE_TYPES = sorted(raw_df["Storage type"].dropna().unique().tolist())
    TOUCH_OPTIONS = sorted(raw_df["Touch"].dropna().unique().tolist())
    CPUS = sorted(raw_df["CPU"].dropna().unique().tolist())
    GPUS = sorted(raw_df["GPU"].dropna().unique().tolist())
    MODELS = sorted(raw_df["Model"].dropna().unique().tolist())
else:
    BRANDS = ["Asus", "Hp", "Lenovo", "Msi", "Apple", "Acer", "Dell", "Razer", "Samsung"]
    STATUSES = ["New", "Refurbished"]
    STORAGE_TYPES = ["SSD", "eMMC"]
    TOUCH_OPTIONS = ["No", "Yes"]
    CPUS = ["Intel Core i7", "Intel Core i5", "Intel Core i3", "Intel Core i9", "AMD Ryzen 7", "AMD Ryzen 5", "Apple M1", "Apple M2"]
    GPUS = ["RTX 3050", "RTX 4060", "RTX 4050", "RTX 3060", "RTX 4070", "RTX 3070", "Integrated"]
    MODELS = ["ExpertBook", "15S", "Katana", "VivoBook", "ThinkPad", "IdeaPad", "Victus", "ROG", "V15"]

@app.route("/")
def home():
    return render_template(
        "index.html",
        brands=BRANDS,
        statuses=STATUSES,
        storage_types=STORAGE_TYPES,
        touch_options=TOUCH_OPTIONS,
        cpus=CPUS,
        gpus=GPUS,
        models=MODELS
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        laptop_name = request.form.get("Laptop")
        if not laptop_name or laptop_name.strip() == "":
            laptop_name = f"{request.form.get('Brand')} {request.form.get('Model')} {request.form.get('CPU')}"

        data = {
            "Laptop": laptop_name,
            "Status": request.form.get("Status", "New"),
            "Brand": request.form.get("Brand", "Asus"),
            "Model": request.form.get("Model", "ExpertBook"),
            "CPU": request.form.get("CPU", "Intel Core i5"),
            "RAM": float(request.form.get("RAM", 8)),
            "Storage": float(request.form.get("Storage", 512)),
            "Storage type": request.form.get("StorageType", "SSD"),
            "GPU": request.form.get("GPU", "RTX 3050"),
            "Screen": float(request.form.get("Screen", 15.6)),
            "Touch": request.form.get("Touch", "No")
        }

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        predicted_price = round(float(pred), 2)

        return render_template(
            "index.html",
            prediction=predicted_price,
            form_data=data,
            brands=BRANDS,
            statuses=STATUSES,
            storage_types=STORAGE_TYPES,
            touch_options=TOUCH_OPTIONS,
            cpus=CPUS,
            gpus=GPUS,
            models=MODELS
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            brands=BRANDS,
            statuses=STATUSES,
            storage_types=STORAGE_TYPES,
            touch_options=TOUCH_OPTIONS,
            cpus=CPUS,
            gpus=GPUS,
            models=MODELS
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
