"""
Data Processing Utility Module for Weather Pattern Visualizer.
Handles CSV ingestion, fuzzy column detection, data cleaning, date parsing,
summary metrics calculation, and automatic statistical insight generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any


COLUMN_ALIASES = {
    "date": ["date", "dt", "datetime", "time", "timestamp", "day", "observation_date"],
    "city": ["city", "location", "station", "place", "city_name", "region"],
    "temperature": ["temperature", "temp", "t_avg", "avg_temp", "mean_temp", "tavg", "temperature_avg"],
    "temp_max": ["temp_max", "max_temp", "temperature_max", "high_temp", "tmax", "max"],
    "temp_min": ["temp_min", "min_temp", "temperature_min", "low_temp", "tmin", "min"],
    "humidity": ["humidity", "rh", "relative_humidity", "hum", "humidity_pct"],
    "rainfall": ["rainfall", "rain", "precipitation", "precip", "prcp", "rainfall_mm", "precip_mm"],
    "wind_speed": ["wind_speed", "windspeed", "wind", "speed", "ws", "wind_kmh", "wind_mps"]
}


def detect_and_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently inspects DataFrame headers and returns a mapping dictionary 
    from standardized key to original column name in the uploaded dataset.
    """
    mapping = {key: None for key in COLUMN_ALIASES}
    
    # Clean lower-cased headers for matching
    df_cols = {col: col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns}
    
    for std_key, aliases in COLUMN_ALIASES.items():
        for original_col, clean_col in df_cols.items():
            if clean_col in aliases:
                mapping[std_key] = original_col
                break
                
    # Fallback partial matching if exact alias not found
    for std_key, aliases in COLUMN_ALIASES.items():
        if mapping[std_key] is None:
            for original_col, clean_col in df_cols.items():
                for alias in aliases:
                    if alias in clean_col and original_col not in mapping.values():
                        mapping[std_key] = original_col
                        break
                if mapping[std_key]:
                    break
                    
    return mapping


def clean_and_transform_data(
    df: pd.DataFrame, 
    column_mapping: Dict[str, Optional[str]], 
    temp_unit: str = "C"
) -> Tuple[pd.DataFrame, list]:
    """
    Cleans, parses, coerces types, deduplicates, and derives date metrics.
    Returns cleaned DataFrame and a list of cleaning warnings/notes.
    """
    cleaned_df = pd.DataFrame()
    warnings = []
    
    # Filter out unmapped keys
    active_mapping = {v: k for k, v in column_mapping.items() if v is not None}
    
    if not active_mapping:
        raise ValueError("No standard weather columns could be mapped from the dataset.")
        
    cleaned_df = df[list(active_mapping.keys())].copy()
    cleaned_df.rename(columns=active_mapping, inplace=True)
    
    # 1. Date parsing
    if "date" in cleaned_df.columns:
        cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=["date"]).copy()
        dropped_dates = initial_len - len(cleaned_df)
        if dropped_dates > 0:
            warnings.append(f"Dropped {dropped_dates} rows with invalid date formats.")
        cleaned_df.sort_values(by="date", inplace=True)
        cleaned_df.reset_index(drop=True, inplace=True)
    else:
        raise ValueError("Dataset missing a recognizable Date column.")

    # 2. Convert Numeric columns
    numeric_cols = ["temperature", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed"]
    for col in numeric_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    # 3. Deduplication
    dedup_subset = ["date"]
    if "city" in cleaned_df.columns:
        dedup_subset.append("city")
    
    dups = cleaned_df.duplicated(subset=dedup_subset).sum()
    if dups > 0:
        cleaned_df.drop_duplicates(subset=dedup_subset, keep="first", inplace=True)
        warnings.append(f"Removed {dups} duplicate records.")

    # 4. Temperature Unit Conversion if Fahrenheit requested
    if temp_unit == "F":
        for t_col in ["temperature", "temp_max", "temp_min"]:
            if t_col in cleaned_df.columns:
                cleaned_df[t_col] = (cleaned_df[t_col] * 9/5) + 32

    # 5. Derive Calendar fields
    cleaned_df["year"] = cleaned_df["date"].dt.year
    cleaned_df["month"] = cleaned_df["date"].dt.month
    cleaned_df["month_name"] = cleaned_df["date"].dt.strftime("%b")
    cleaned_df["month_year"] = cleaned_df["date"].dt.strftime("%Y-%m")
    cleaned_df["day"] = cleaned_df["date"].dt.day
    cleaned_df["day_of_week"] = cleaned_df["date"].dt.strftime("%a")
    
    # Derive temperature average if missing but min & max exist
    if "temperature" not in cleaned_df.columns and "temp_max" in cleaned_df.columns and "temp_min" in cleaned_df.columns:
        cleaned_df["temperature"] = (cleaned_df["temp_max"] + cleaned_df["temp_min"]) / 2
        warnings.append("Derived average temperature from min and max temperature values.")

    # Derive rainy day flag
    if "rainfall" in cleaned_df.columns:
        cleaned_df["is_rainy"] = cleaned_df["rainfall"] > 0.1

    return cleaned_df, warnings


def calculate_summary_metrics(
    df: pd.DataFrame, 
    selected_city: Optional[str] = None, 
    temp_unit: str = "C"
) -> Dict[str, Any]:
    """
    Computes overview metric values from currently filtered dataset.
    Handles missing metrics gracefully without fabricating data.
    """
    filtered_df = df.copy()
    if selected_city and "city" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["city"] == selected_city]

    unit_symbol = "°F" if temp_unit == "F" else "°C"
    
    metrics = {
        "unit_symbol": unit_symbol,
        "total_records": len(filtered_df),
        "date_start": filtered_df["date"].min().strftime("%Y-%m-%d") if not filtered_df.empty else "N/A",
        "date_end": filtered_df["date"].max().strftime("%Y-%m-%d") if not filtered_df.empty else "N/A",
        "avg_temp": None,
        "max_temp": None,
        "min_temp": None,
        "total_rainfall": None,
        "avg_humidity": None,
        "avg_wind_speed": None,
        "rainy_days": None
    }

    if filtered_df.empty:
        return metrics

    if "temperature" in filtered_df.columns and filtered_df["temperature"].notna().any():
        metrics["avg_temp"] = round(filtered_df["temperature"].mean(), 1)
        
    if "temp_max" in filtered_df.columns and filtered_df["temp_max"].notna().any():
        metrics["max_temp"] = round(filtered_df["temp_max"].max(), 1)
    elif "temperature" in filtered_df.columns and filtered_df["temperature"].notna().any():
        metrics["max_temp"] = round(filtered_df["temperature"].max(), 1)

    if "temp_min" in filtered_df.columns and filtered_df["temp_min"].notna().any():
        metrics["min_temp"] = round(filtered_df["temp_min"].min(), 1)
    elif "temperature" in filtered_df.columns and filtered_df["temperature"].notna().any():
        metrics["min_temp"] = round(filtered_df["temperature"].min(), 1)

    if "rainfall" in filtered_df.columns and filtered_df["rainfall"].notna().any():
        metrics["total_rainfall"] = round(filtered_df["rainfall"].sum(), 1)
        metrics["rainy_days"] = int((filtered_df["rainfall"] > 0.1).sum())

    if "humidity" in filtered_df.columns and filtered_df["humidity"].notna().any():
        metrics["avg_humidity"] = round(filtered_df["humidity"].mean(), 1)

    if "wind_speed" in filtered_df.columns and filtered_df["wind_speed"].notna().any():
        metrics["avg_wind_speed"] = round(filtered_df["wind_speed"].mean(), 1)

    return metrics


def generate_automated_narrative(df: pd.DataFrame, temp_unit: str = "C") -> Dict[str, str]:
    """
    Generates data-backed statistical text insights for Temperature, Rainfall, and Humidity trends.
    Avoids guessing or fabricating statements when data is missing.
    """
    unit = "°F" if temp_unit == "F" else "°C"
    narratives = {}

    if df.empty:
        return {"temp_narrative": "No data available for trend analysis.", 
                "rain_narrative": "No rainfall data available.", 
                "humidity_narrative": "No humidity data available."}

    # 1. Temperature Narrative
    if "temperature" in df.columns and len(df) > 1:
        avg_temp = df["temperature"].mean()
        min_temp = df["temperature"].min()
        max_temp = df["temperature"].max()
        
        # Monthly highest/lowest
        monthly_avg = df.groupby("month_name", observed=False)["temperature"].mean()
        warmest_month = monthly_avg.idxmax() if not monthly_avg.empty else "N/A"
        coolest_month = monthly_avg.idxmin() if not monthly_avg.empty else "N/A"
        
        # Linear slope check
        days_diff = (df["date"].max() - df["date"].min()).days
        if days_diff > 10:
            first_half = df.iloc[: len(df)//2]["temperature"].mean()
            second_half = df.iloc[len(df)//2 :]["temperature"].mean()
            diff = second_half - first_half
            if abs(diff) < 0.5:
                trend_str = "remained overall stable"
            elif diff > 0:
                trend_str = f"increased by approximately {diff:.1f}{unit} over the observed timeframe"
            else:
                trend_str = f"decreased by approximately {abs(diff):.1f}{unit} over the observed timeframe"
        else:
            trend_str = "shows short-term daily fluctuations"

        narratives["temp_narrative"] = (
            f"Over the selected period ({len(df)} records), average temperature was **{avg_temp:.1f}{unit}** "
            f"(ranging from **{min_temp:.1f}{unit}** to **{max_temp:.1f}{unit}**). "
            f"The data indicates that temperatures {trend_str}. "
            f"The warmest month on average was **{warmest_month}**, while **{coolest_month}** recorded the lowest monthly average."
        )
    else:
        narratives["temp_narrative"] = "Insufficient temperature observations to extract trends."

    # 2. Rainfall Narrative
    if "rainfall" in df.columns and df["rainfall"].notna().any():
        total_rain = df["rainfall"].sum()
        rainy_days = (df["rainfall"] > 0.1).sum()
        max_rain_day = df.loc[df["rainfall"].idxmax()]
        
        monthly_rain = df.groupby("month_name", observed=False)["rainfall"].sum()
        wettest_m = monthly_rain.idxmax() if not monthly_rain.empty else "N/A"
        driest_m = monthly_rain.idxmin() if not monthly_rain.empty else "N/A"

        narratives["rain_narrative"] = (
            f"Total recorded rainfall was **{total_rain:.1f} mm** across **{rainy_days} rainy days** "
            f"({(rainy_days/len(df))*100:.1f}% of observed period). "
            f"The single wettest day was **{max_rain_day['date'].strftime('%Y-%m-%d')}** with **{max_rain_day['rainfall']:.1f} mm**. "
            f"On a seasonal basis, **{wettest_m}** was the wettest month and **{driest_m}** was the driest."
        )
    else:
        narratives["rain_narrative"] = "Rainfall data is not available in the active dataset. Upload a dataset containing a rainfall column to unlock precipitation insights."

    # 3. Humidity & Wind Narrative
    hum_part, wind_part = "", ""
    if "humidity" in df.columns and df["humidity"].notna().any():
        avg_hum = df["humidity"].mean()
        hum_part = f"Average relative humidity was **{avg_hum:.1f}%**. "
        if "temperature" in df.columns:
            corr = df["temperature"].corr(df["humidity"])
            if not np.isnan(corr):
                relation = "inverse (as temperature rises, humidity tends to drop)" if corr < -0.2 else ("positive" if corr > 0.2 else "weak/neutral")
                hum_part += f"The correlation between temperature and humidity is **{relation}** (r = {corr:.2f}). "

    if "wind_speed" in df.columns and df["wind_speed"].notna().any():
        avg_w = df["wind_speed"].mean()
        max_w = df["wind_speed"].max()
        wind_part = f"Average wind speed recorded was **{avg_w:.1f} km/h**, peaking at **{max_w:.1f} km/h**."

    narratives["humidity_narrative"] = (hum_part + wind_part) if (hum_part or wind_part) else "Humidity and wind data are not present in this dataset."

    return narratives
