import pandas as pd
import numpy as np
import os

print("🌾 Generating realistic hybrid pest prediction dataset...")
np.random.seed(42)

rows_per_class = 34  # ~102 total rows, balanced across 3 risk levels
data = []

def add_noise(val, scale=0.02):
    """Add small Gaussian noise to avoid identical rows."""
    return val + np.random.normal(0, scale)

# ── HIGH RISK (~34 rows) ──
# ndvi < 0.4, humidity > 70, rainfall > 10
for _ in range(rows_per_class):
    ndvi = np.clip(add_noise(np.random.uniform(0.1, 0.38), 0.03), 0.1, 0.8)
    temp = np.clip(add_noise(np.random.uniform(29, 35), 0.5), 24, 35)
    humidity = np.clip(add_noise(np.random.uniform(72, 90), 2.0), 40, 90)
    rainfall = np.clip(add_noise(np.random.uniform(12, 25), 1.5), 0, 25)
    pest = np.random.choice(["Aphids", "Leaf Blight"])
    data.append([round(ndvi, 2), round(temp, 1), round(humidity, 1),
                 round(rainfall, 1), "High", pest])

# ── MEDIUM RISK (~34 rows) ──
# ndvi 0.4–0.7, moderate humidity
for _ in range(rows_per_class):
    ndvi = np.clip(add_noise(np.random.uniform(0.4, 0.68), 0.03), 0.1, 0.8)
    temp = np.clip(add_noise(np.random.uniform(26, 33), 0.5), 24, 35)
    humidity = np.clip(add_noise(np.random.uniform(50, 72), 2.0), 40, 90)
    rainfall = np.clip(add_noise(np.random.uniform(4, 14), 1.0), 0, 25)
    pest = np.random.choice(["Armyworm", "Aphids"])
    data.append([round(ndvi, 2), round(temp, 1), round(humidity, 1),
                 round(rainfall, 1), "Medium", pest])

# ── LOW RISK (~34 rows) ──
# ndvi > 0.7, low humidity, low rainfall
for _ in range(rows_per_class):
    ndvi = np.clip(add_noise(np.random.uniform(0.7, 0.8), 0.02), 0.1, 0.8)
    temp = np.clip(add_noise(np.random.uniform(24, 30), 0.5), 24, 35)
    humidity = np.clip(add_noise(np.random.uniform(40, 55), 2.0), 40, 90)
    rainfall = np.clip(add_noise(np.random.uniform(0, 5), 0.5), 0, 25)
    pest = "No Pest"
    data.append([round(ndvi, 2), round(temp, 1), round(humidity, 1),
                 round(rainfall, 1), "Low", pest])

df = pd.DataFrame(data, columns=["ndvi", "temperature", "humidity",
                                  "rainfall", "risk_level", "pest_type"])

# Shuffle to remove ordering bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

csv_path = os.path.join(os.path.dirname(__file__), "pest_dataset.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Dataset saved to {csv_path} — {len(df)} rows")
print(f"\n📊 Risk level distribution:\n{df['risk_level'].value_counts().to_string()}")
print(f"\n🐛 Pest type distribution:\n{df['pest_type'].value_counts().to_string()}")
