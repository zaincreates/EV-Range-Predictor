import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

battery_pct   = np.random.uniform(10, 100, n)
speed_kmh     = np.random.uniform(20, 130, n)
temperature_c = np.random.uniform(-5, 45, n)
passengers    = np.random.randint(1, 6, n)

temp_penalty  = np.abs(temperature_c - 22) * 0.4
range_km = (
    battery_pct * 3.5
    - speed_kmh * 0.6
    - temp_penalty
    - passengers * 4
    + np.random.normal(0, 8, n)
)
range_km = np.clip(range_km, 5, 400)

df = pd.DataFrame({
    'battery_pct': battery_pct,
    'speed_kmh': speed_kmh,
    'temperature_c': temperature_c,
    'passengers': passengers,
    'range_km': range_km
})

df.to_csv('ev_data.csv', index=False)
print("Dataset saved! Shape:", df.shape)
print(df.head())
