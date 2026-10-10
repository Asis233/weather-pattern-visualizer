"""
Machine Learning Temperature Prediction Module for Weather Pattern Visualizer.
Uses Scikit-learn Linear / Polynomial Regression to compute temperature trends 
and educational short-term statistical forecasts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as px_go
from utils.visualizations import PASTEL_COLORS, apply_plotly_theme

def train_temperature_trend_model(
    df: pd.DataFrame, 
    forecast_days: int = 30, 
    use_polynomial: bool = True,
    temp_unit: str = "C"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains a simple regression model on historical dates to project future temperature trends.
    
    Returns:
    - Dict of model evaluation metrics (R2, MAE, RMSE, trend slope)
    - Combined DataFrame containing actual historical points and predicted trendline + forecast points
    """
    if "temperature" not in df.columns or df["temperature"].dropna().count() < 15:
        raise ValueError("Minimum 15 historical temperature observations required for trend modeling.")

    clean_df = df[["date", "temperature"]].dropna().sort_values("date").copy()
    
    # Feature Engineering: Days relative to start date
    start_date = clean_df["date"].min()
    clean_df["days_since_start"] = (clean_df["date"] - start_date).dt.days
    
    X = clean_df[["days_since_start"]].values
    y = clean_df["temperature"].values
    
    # Fit Model
    if use_polynomial and len(clean_df) >= 30:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_trans = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_trans, y)
        y_pred = model.predict(X_trans)
    else:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

    # Compute Evaluation Metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Historical Fitted Output
    clean_df["predicted_temp"] = y_pred
    clean_df["data_type"] = "Historical Actual"
    
    # Generate Future Date Projections
    last_date = clean_df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_days = (future_dates - start_date).days.values.reshape(-1, 1)
    
    if use_polynomial and len(clean_df) >= 30:
        future_days_trans = poly.transform(future_days)
        future_preds = model.predict(future_days_trans)
    else:
        future_preds = model.predict(future_days)
        
    future_df = pd.DataFrame({
        "date": future_dates,
        "temperature": np.nan,
        "predicted_temp": future_preds,
        "days_since_start": (future_dates - start_date).days,
        "data_type": "Statistical Forecast"
    })
    
    combined_df = pd.concat([clean_df, future_df], ignore_index=True)
    
    # Overall linear slope direction
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    slope_per_year = linear_model.coef_[0] * 365.25
    unit = "°F" if temp_unit == "F" else "°C"
    
    metrics = {
        "r2_score": round(r2, 3),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "total_training_samples": len(clean_df),
        "forecast_days": forecast_days,
        "model_type": "Polynomial Regression (Degree 2)" if (use_polynomial and len(clean_df) >= 30) else "Linear Regression",
        "trend_rate": f"{slope_per_year:+.2f}{unit}/year"
    }
    
    return metrics, combined_df


def plot_temperature_prediction(combined_df: pd.DataFrame, temp_unit: str = "C"):
    """
    Creates a Plotly chart showing actual observations vs regression trend & future projection.
    """
    unit = "°F" if temp_unit == "F" else "°C"
    fig = px_go.Figure()

    actual_df = combined_df[combined_df["data_type"] == "Historical Actual"]
    forecast_df = combined_df[combined_df["data_type"] == "Statistical Forecast"]

    # Historical Actual scatter/line
    fig.add_trace(px_go.Scatter(
        x=actual_df["date"],
        y=actual_df["temperature"],
        mode="markers+lines",
        name=f"Historical Actual ({unit})",
        marker=dict(size=5, color=PASTEL_COLORS["blue_main"], opacity=0.6),
        line=dict(color=PASTEL_COLORS["blue_light"], width=1.5),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.1f}" + unit + "<extra></extra>"
    ))

    # Fitted Trendline
    fig.add_trace(px_go.Scatter(
        x=actual_df["date"],
        y=actual_df["predicted_temp"],
        mode="lines",
        name="Fitted Trendline",
        line=dict(color=PASTEL_COLORS["lavender_main"], width=2.5),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Trend Fit:</b> %{y:.1f}" + unit + "<extra></extra>"
    ))

    # Forecast Projection
    fig.add_trace(px_go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_temp"],
        mode="lines",
        name=f"Projected Forecast ({len(forecast_df)} Days)",
        line=dict(color=PASTEL_COLORS["peach_main"], width=3, dash="dash"),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast:</b> %{y:.1f}" + unit + "<extra></extra>"
    ))

    apply_plotly_theme(fig, "Temperature Trend Projection & Short-Term Forecast", "Date", f"Temperature ({unit})")
    return fig
