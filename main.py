from fastapi import FastAPI, Header, HTTPException
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

app = FastAPI()

API_KEY = os.getenv("ML_SERVICE_API_KEY", "test_key_123")

@app.get("/")
def home():
    return {"message": "ML Service Running"}

@app.post("/predict")
async def predict(data: dict, x_api_key: str = Header(None)):
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ---- INPUTS ----
    ndvi = data.get("ndvi", 0.5)
    temperature = data.get("temperature", 30)
    humidity = data.get("humidity", 60)
    rainfall = data.get("rainfall", 0)
    crop_type = data.get("cropType", "unknown")

    # ---- SIMPLE MODEL LOGIC ----
    # (This is your baseline ML logic — acceptable for project)

    risk_score = 0

    if ndvi < 0.4:
        risk_score += 2
    elif ndvi < 0.6:
        risk_score += 1

    if temperature > 35:
        risk_score += 2
    elif temperature > 30:
        risk_score += 1

    if humidity > 80:
        risk_score += 2
    elif humidity > 65:
        risk_score += 1

    if rainfall > 50:
        risk_score += 1

    # ---- CLASSIFICATION ----
    if risk_score >= 5:
        risk = "High"
    elif risk_score >= 3:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "risk": risk,
        "risk_score": risk_score,
        "inputs": {
            "ndvi": ndvi,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "cropType": crop_type
        }
    }