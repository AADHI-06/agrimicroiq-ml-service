from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import os
import uvicorn
import joblib
import numpy as np
import firebase_admin
from firebase_admin import auth, credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Firebase Admin
cred = None
service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

if service_account_json:
    try:
        import json
        cert_dict = json.loads(service_account_json)
        cred = credentials.Certificate(cert_dict)
    except Exception as e:
        print(f"Warning: Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}")

if not cred:
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "firebaseServiceAccount.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)

if not firebase_admin._apps:
    if cred:
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()

from typing import Optional

security = HTTPBearer(auto_error=False)
ML_SECURITY_ENABLED = os.getenv("ML_SECURITY_ENABLED", "true").lower() == "true"

async def verify_firebase_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not ML_SECURITY_ENABLED:
        return {"uid": "dev_user", "email": "dev@example.com", "email_verified": True}
        
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing")
        
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        if not decoded_token.get("email_verified"):
            raise HTTPException(status_code=403, detail="Email not verified")
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

app = FastAPI()

# Phase 74: Stabilize Integration (CORS & Security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://agri-micro-iq.web.app",
        "https://agrimicroiq-app.onrender.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEY = os.getenv("ML_SERVICE_API_KEY", "test_key_123")
BASE_DIR = os.path.dirname(__file__)

# ── Load Models ──
def load_pickle(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Warning: Could not load {path}. Error: {e}")
        return None

model = load_pickle(os.path.join(BASE_DIR, "pest_model.pkl"))
label_encoders = load_pickle(os.path.join(BASE_DIR, "label_encoders.pkl"))
yield_model = load_pickle(os.path.join(BASE_DIR, "yield_model.pkl"))

# ── Pydantic Schema ──
class PestPredictionInput(BaseModel):
    ndvi: float = Field(..., description="Normalized vegetation index (0.1–0.8)")
    temperature: float = Field(..., description="Temperature in °C (24–35)")
    humidity: float = Field(..., description="Humidity in % (40–90)")
    rainfall: float = Field(..., description="Rainfall in mm (0–25)")

@app.get("/")
def home():
    return {"message": "AgriMicro IQ Native ML Service Alive"}

@app.post("/predict-pest")
async def predict_pest(data: PestPredictionInput, user: dict = Depends(verify_firebase_token)):
    if model is None or label_encoders is None:
        raise HTTPException(status_code=500, detail="Model or label encoders not loaded.")

    features = np.array([[data.ndvi, data.temperature, data.humidity, data.rainfall]])
    try:
        prediction = model.predict(features)[0]
        risk_level = label_encoders["risk_level"].inverse_transform([prediction[0]])[0]
        pest_type = label_encoders["pest_type"].inverse_transform([prediction[1]])[0]
        
        proba_risk = max(model.estimators_[0].predict_proba(features)[0])
        proba_pest = max(model.estimators_[1].predict_proba(features)[0])
        probability = round(float(max(proba_risk, proba_pest)), 2)

        return {"risk_level": risk_level, "pest_type": pest_type, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: dict, user: dict = Depends(verify_firebase_token)):
    if model is None or label_encoders is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        ndvi = float(data.get("ndvi", 0.5))
        temp = float(data.get("temperature", 30.0))
        hum = float(data.get("humidity", 60.0))
        rain = float(data.get("rainfall", 0.0))
        
        features = [[ndvi, temp, hum, rain]]
        prediction = model.predict(features)[0]
        
        risk_level = label_encoders["risk_level"].inverse_transform([prediction[0]])[0]
        pest_type = label_encoders["pest_type"].inverse_transform([prediction[1]])[0]
        prob = round(float(max(model.estimators_[0].predict_proba(features)[0])), 2)
        
        return {
            "riskLevel": risk_level,
            "probability": prob,
            "predictedPest": pest_type,
            "inputs": {"ndvi": ndvi, "temperature": temp, "humidity": hum, "rainfall": rain}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

CROP_METRICS = {
    "rice": {"base": 4.0, "price": 20000},
    "wheat": {"base": 3.5, "price": 18000},
    "maize": {"base": 5.0, "price": 17000},
    "corn": {"base": 5.0, "price": 17000},
    "cotton": {"base": 2.0, "price": 60000},
    "soybean": {"base": 2.5, "price": 45000},
    "barley": {"base": 3.0, "price": 22000}
}

@app.post("/simulate-yield")
async def simulate_yield(data: dict, user: dict = Depends(verify_firebase_token)):
    crop_type = data.get("cropType", "wheat").lower().strip()
    if crop_type not in CROP_METRICS:
        raise HTTPException(status_code=400, detail=f"Unsupported crop: {crop_type}")

    ndvi = float(data.get("avg_ndvi", 0.6))
    water = float(data.get("avg_water", 100.0))
    fert = float(data.get("avg_fertilizer", 25.0))
    temp = float(data.get("temperature", 24.5))

    def calculate(in_ndvi, in_water, in_fert, in_temp):
        nf = np.clip(in_ndvi / 0.8, 0.5, 1.2)
        wf = np.clip(in_water / 120.0, 0.7, 1.2)
        ff = np.clip(in_fert / 30.0, 0.7, 1.2)
        tf = np.clip(1 - (abs(in_temp - 25.0) / 20.0), 0.6, 1.1)
        
        y = round(CROP_METRICS[crop_type]["base"] * nf * wf * ff * tf, 2)
        rev = y * CROP_METRICS[crop_type]["price"]
        cost = (in_water * 2) + (in_fert * 50)
        return {"yield": y, "profit": round(rev - cost, 2), "factors": {"ndvi": round(float(nf), 2), "water": round(float(wf), 2), "fert": round(float(ff), 2), "temp": round(float(tf), 2)}}

    opt = calculate(ndvi, water, fert, temp)
    cur = calculate(ndvi * 0.85, water * 0.7, fert * 0.7, temp)

    return {
        "cropType": crop_type,
        "optimized": {"yield": f"{opt['yield']} tons", "profit": f"₹{int(opt['profit']):,}", "factors": opt["factors"]},
        "current": {"yield": f"{cur['yield']} tons", "profit": f"₹{int(cur['profit']):,}"},
        "gain_pct": round(((opt['yield'] - cur['yield']) / cur['yield']) * 100, 1) if cur['yield'] > 0 else 0
    }

@app.post("/optimize-resource")
async def optimize_resource(data: dict, user: dict = Depends(verify_firebase_token)):
    ndvi = float(data.get("ndvi", 0.5))
    pest = data.get("pest_prediction", "None").upper()
    fert, water = 25.0, 100.0
    
    if ndvi < 0.4:
        fert += 15.0
        water += 40.0
    elif ndvi > 0.7:
        fert -= 5.0
    if pest in ["LEAF BLIGHT", "HIGH"]:
        water += 20.0
        
    return {"fertilizerAmount": round(fert, 1), "waterRequirement": round(water, 1)}

if __name__ == "__main__":
    # Render maps standard PORT env var automatically
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)