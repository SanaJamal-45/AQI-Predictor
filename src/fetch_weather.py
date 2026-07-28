import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

print("API Key:", API_KEY)
print("Length:", len(API_KEY) if API_KEY else "Not found")

CITY = "Karachi"

url = (
    f"https://api.openweathermap.org/data/2.5/weather"
    f"?q={CITY}&appid={API_KEY}&units=metric"
)

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind_speed = data["wind"]["speed"]
    city = data["name"]
    country = data["sys"]["country"]
    print(city)
    print(country)
    print(temperature)
    print(humidity)
    print(pressure)
    print(wind_speed)
    current_time = datetime.now()
    df = pd.DataFrame([{
    "datetime": current_time,
    "city": city,
    "country": country,
    "temperature": temperature,
    "humidity": humidity,
    "pressure": pressure,
    "wind_speed": wind_speed
}])

    print(df)
    df.to_csv("data/raw/weather.csv", index=False)
else:
    print("Error:", response.status_code, response.text)