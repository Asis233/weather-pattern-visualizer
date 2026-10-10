"""
End-to-End Test Suite for Weather Pattern Visualizer
Verifies data processing, chart generation, metrics calculation, ML trend prediction, and live weather.
"""

import sys
import io

# Enforce UTF-8 for console output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from utils.data_processing import detect_and_map_columns, clean_and_transform_data, calculate_summary_metrics, generate_automated_narrative
from utils.visualizations import (
    plot_daily_temperature_trend, plot_monthly_avg_temperature, plot_min_max_temp_range,
    plot_daily_rainfall, plot_monthly_rainfall, plot_humidity_wind_trends,
    plot_temp_vs_humidity_scatter, plot_rainfall_vs_humidity_scatter, plot_city_comparison_bar,
    plot_seaborn_temp_distribution, plot_seaborn_monthly_boxplot, plot_seaborn_correlation_heatmap
)
from utils.ml_prediction import train_temperature_trend_model, plot_temperature_prediction
from utils.weather_api import fetch_current_weather, detect_user_location

def run_tests():
    print("--- 1. Testing Single City Sample Dataset ---")
    df_raw = pd.read_csv("data/sample_weather.csv")
    mapping = detect_and_map_columns(df_raw)
    assert mapping["date"] == "date", "Date mapping failed"
    assert mapping["temperature"] == "temperature", "Temperature mapping failed"
    
    df, warnings = clean_and_transform_data(df_raw, mapping, temp_unit="C")
    assert len(df) == 366, f"Expected 366 rows, got {len(df)}"
    print("[PASS] Single city data cleaning passed. Warnings:", warnings)

    metrics = calculate_summary_metrics(df, temp_unit="C")
    assert metrics["avg_temp"] is not None, "Avg temp calculation failed"
    assert metrics["total_rainfall"] is not None, "Rainfall calculation failed"
    print("[PASS] Metrics calculation passed:", metrics)

    narratives = generate_automated_narrative(df, temp_unit="C")
    assert "temp_narrative" in narratives, "Narrative generation failed"
    print("[PASS] Narrative generation passed.")

    # Visualizations
    fig1 = plot_daily_temperature_trend(df, temp_unit="C")
    fig2 = plot_monthly_avg_temperature(df, temp_unit="C")
    fig3 = plot_min_max_temp_range(df, temp_unit="C")
    fig4 = plot_daily_rainfall(df)
    fig5 = plot_monthly_rainfall(df)
    fig6 = plot_humidity_wind_trends(df)
    fig7 = plot_temp_vs_humidity_scatter(df, temp_unit="C")
    fig8 = plot_rainfall_vs_humidity_scatter(df)
    fig9 = plot_seaborn_temp_distribution(df, temp_unit="C")
    fig10 = plot_seaborn_monthly_boxplot(df, temp_unit="C")
    fig11 = plot_seaborn_correlation_heatmap(df)
    print("[PASS] All 11 visualizers generated successfully without errors!")

    print("\n--- 2. Testing Multi-City Dataset ---")
    df_multi_raw = pd.read_csv("data/sample_multi_city.csv")
    mapping_multi = detect_and_map_columns(df_multi_raw)
    df_multi, _ = clean_and_transform_data(df_multi_raw, mapping_multi, temp_unit="C")
    fig_comp = plot_city_comparison_bar(df_multi, "temperature", "Average Temperature", temp_unit="C")
    assert fig_comp is not None, "City comparison chart failed"
    print("[PASS] Multi-city comparison chart generated successfully!")

    print("\n--- 3. Testing ML Trend Prediction ---")
    ml_metrics, pred_df = train_temperature_trend_model(df, forecast_days=30, use_polynomial=True, temp_unit="C")
    assert ml_metrics["r2_score"] is not None, "ML R2 score missing"
    fig_ml = plot_temperature_prediction(pred_df, temp_unit="C")
    print("[PASS] ML model training & trend prediction passed:", ml_metrics)

    print("\n--- 4. Testing Real-Time Live Weather & IP Location Auto-Detection ---")
    loc_res = detect_user_location()
    print("[PASS] IP GeoLocation Auto-Detection Result:", loc_res.get("city"), loc_res.get("country"))
    
    api_res = fetch_current_weather("London")
    assert api_res["success"] == True, f"Live weather request failed: {api_res}"
    print(f"[PASS] Live weather fetch passed ({api_res['city']}, {api_res['temp']}{api_res['units']}, {api_res['description']}) via {api_res['provider']}")

    print("\n[SUCCESS] ALL E2E TESTS PASSED 100%!")

if __name__ == "__main__":
    run_tests()
