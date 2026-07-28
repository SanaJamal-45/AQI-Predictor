import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.9056
LON = 67.0822

url = (
    f"https://api.openweathermap.org/data/2.5/air_pollution"
    f"?lat={LAT}&lon={LON}&appid={API_KEY}"
)

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    aqi = data["list"][0]["main"]["aqi"]

    co = data["list"][0]["components"]["co"]
    no2 = data["list"][0]["components"]["no2"]
    o3 = data["list"][0]["components"]["o3"]
    pm2_5 = data["list"][0]["components"]["pm2_5"]
    pm10 = data["list"][0]["components"]["pm10"]

    current_time = datetime.now()

    df = pd.DataFrame([{
        "datetime": current_time,
        "aqi": aqi,
        "pm2_5": pm2_5,
        "pm10": pm10,
        "co": co,
        "no2": no2,
        "o3": o3
    }])

    print(df)
    df.to_csv("data/raw/air_quality.csv", index=False)
else:
    print("Error:", response.status_code, response.text)