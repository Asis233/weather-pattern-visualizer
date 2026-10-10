"""
Script to generate sample historical weather datasets for Weather Pattern Visualizer.
Generates realistic, physically plausible seasonal variations for temperature, humidity,
rainfall, and wind speed, spanning up to the CURRENT DATE.
"""

import os
import datetime
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

# Generate dates up to TODAY'S CURRENT DATE
today = datetime.date.today()
start_date = today - datetime.timedelta(days=365)
dates = pd.date_range(start=start_date, end=today, freq="D")
n_days = len(dates)

# 1. Single City Sample Dataset (Seattle)
np.random.seed(42)

# Seasonal temperature wave (colder in Jan/Dec, warmer in July/Aug)
day_of_year = dates.dayofyear
temp_base = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
temp_noise = np.random.normal(0, 2.5, n_days)
temperature = np.round(temp_base + temp_noise, 1)

temp_max = np.round(temperature + np.random.uniform(2.5, 6.0, n_days), 1)
temp_min = np.round(temperature - np.random.uniform(2.5, 6.0, n_days), 1)

# Seasonal rainfall (more rain in winter, drier in summer)
rain_prob = 0.45 - 0.25 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
rain_occurred = np.random.binomial(1, np.clip(rain_prob, 0.1, 0.7), n_days)
rainfall_amount = np.round(rain_occurred * np.random.exponential(scale=7.0, size=n_days), 1)

# Humidity inversely related to temp + boosted on rainy days
humidity_base = 75 - 15 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
humidity = np.clip(np.round(humidity_base + 12 * rain_occurred + np.random.normal(0, 5, n_days)), 30, 98)

# Wind speed
wind_speed = np.clip(np.round(14 + 5 * np.random.normal(0, 1, n_days) + 4 * rain_occurred), 2, 45)

sample_df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "city": "Seattle",
    "temperature": temperature,
    "temp_max": temp_max,
    "temp_min": temp_min,
    "humidity": humidity,
    "rainfall": rainfall_amount,
    "wind_speed": wind_speed
})

sample_df.to_csv("data/sample_weather.csv", index=False)
print(f"Created data/sample_weather.csv with {len(sample_df)} records (spanning {start_date} to {today}).")

# 2. Multi-City Sample Dataset (New York, London, Tokyo)
cities_config = {
    "New York": {"temp_mean": 13, "temp_amp": 14, "rain_scale": 8.0, "rain_prob": 0.35},
    "London": {"temp_mean": 11, "temp_amp": 8, "rain_scale": 5.0, "rain_prob": 0.45},
    "Tokyo": {"temp_mean": 16, "temp_amp": 12, "rain_scale": 11.0, "rain_prob": 0.38}
}

multi_rows = []
for city, cfg in cities_config.items():
    t_base = cfg["temp_mean"] + cfg["temp_amp"] * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    t_noise = np.random.normal(0, 2.2, n_days)
    t_avg = np.round(t_base + t_noise, 1)
    
    t_hi = np.round(t_avg + np.random.uniform(2.0, 5.5, n_days), 1)
    t_lo = np.round(t_avg - np.random.uniform(2.0, 5.5, n_days), 1)
    
    r_occurred = np.random.binomial(1, cfg["rain_prob"], n_days)
    r_amt = np.round(r_occurred * np.random.exponential(scale=cfg["rain_scale"], size=n_days), 1)
    
    hum = np.clip(np.round(68 + 10 * r_occurred + np.random.normal(0, 6, n_days)), 25, 95)
    w_spd = np.clip(np.round(12 + np.random.normal(0, 3.5, n_days)), 3, 40)
    
    for i in range(n_days):
        multi_rows.append({
            "date": dates[i].strftime("%Y-%m-%d"),
            "city": city,
            "temperature": t_avg[i],
            "temp_max": t_hi[i],
            "temp_min": t_lo[i],
            "humidity": hum[i],
            "rainfall": r_amt[i],
            "wind_speed": w_spd[i]
        })

multi_df = pd.DataFrame(multi_rows)
multi_df.to_csv("data/sample_multi_city.csv", index=False)
print(f"Created data/sample_multi_city.csv with {len(multi_df)} records (spanning {start_date} to {today}).")
