# 🤖 AgriMicro IQ – ML Service

This repository contains the **Machine Learning microservice** for the **AgriMicro IQ platform**, responsible for pest risk prediction, yield estimation, and profit calculation using environmental and vegetation data.

---

## 🚀 Overview

The ML service is built using **FastAPI** and provides REST APIs for:

* 🐛 Pest Risk Prediction (Classification)
* 🌾 Yield Estimation (Regression)
* 💰 Profit Estimation (Derived Calculation)

It processes inputs such as NDVI, temperature, humidity, rainfall, and crop-related parameters to generate actionable insights.

---

## 🧠 Machine Learning Models

### 1️⃣ Pest Risk Prediction

* **Type:** Classification
* **Algorithm:** Random Forest Classifier
* **Inputs:**

  * NDVI
  * Temperature
  * Humidity
  * Rainfall
* **Outputs:**

  * Risk Level (Low / Medium / High)
  * Pest Type
  * Probability Score

---

### 2️⃣ Yield Prediction

* **Type:** Regression
* **Algorithm:** Linear Regression
* **Inputs:**

  * NDVI
  * Water Input
  * Fertilizer Input
  * Temperature
* **Output:**

  * Expected Yield

---

### 3️⃣ Profit Estimation

* **Type:** Derived Regression
* **Logic:**

  * Based on predicted yield and crop-specific pricing
* **Output:**

  * Expected Profit

---

## 🛠️ Tech Stack

* Python 3.x
* FastAPI
* Scikit-learn
* Pandas / NumPy
* Uvicorn

---

## 📂 Project Structure

/ml-service
│
├── main.py              # FastAPI entry point
├── models/              # Trained ML models (.pkl files)
├── schemas/             # Request/response schemas
├── services/            # Prediction logic
├── utils/               # Helper functions
├── requirements.txt     # Dependencies
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone Repository

git clone https://github.com/AADHI-06/agrimicroiq-ml-service.git
cd agrimicroiq-ml-service

---

### 2. Install Dependencies

pip install -r requirements.txt

---

### 3. Run the Service

uvicorn main:app --reload

---

### 4. Access API Docs

http://localhost:8000/docs

---

## 🌐 Deployment

| Service    | URL                                         |
| ---------- | ------------------------------------------- |
| ML Service | https://agrimicroiq-ml-service.onrender.com |

---

## 🔌 API Endpoints

### 🔹 Health Check

GET /health

Response:
{
"status": "ok"
}

---

### 🔹 Pest Prediction

POST /predict

Request:
{
"ndvi": 0.5,
"temperature": 28,
"humidity": 70,
"rainfall": 5
}

Response:
{
"risk_level": "Medium",
"pest": "Aphids",
"probability": 0.78
}

---

## 🔐 Security

* API access protected via Authorization header
* Token-based authentication supported
* No sensitive keys exposed in code

---

## 📊 Dataset

* Hybrid dataset (synthetic + realistic patterns)
* Features include:

  * Vegetation indices (NDVI)
  * Weather parameters
  * Crop inputs

---

## 🧪 Testing

* API testing via Swagger UI
* Model validation using sample datasets
* Integration testing with backend

---

## ⚠️ Notes

* Model is optimized for **demo + practical usability**, not full-scale agricultural deployment
* Performance can be improved with real-world datasets

---

## 📌 Future Improvements

* Advanced deep learning models
* Real-time model retraining
* Federated learning enhancement
* Crop-specific adaptive models

---

## 👥 Contributors

* Backend & ML Engineer – Model development, API design
* Frontend Engineer – Integration with ML endpoints

---

## 📄 License

Academic project – for educational use only

---

⭐ Star this repo if you find it useful!
