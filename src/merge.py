import pandas as pd

weather = pd.read_csv("data/raw/weather.csv")

air = pd.read_csv("data/raw/air_quality.csv")

master = pd.concat([weather, air], axis=1)

print(master)

master.to_csv("data/raw/master_dataset.csv", index=False)

print("Master dataset created successfully!")