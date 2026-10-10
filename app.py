"""
Weather Pattern Visualizer - Main Streamlit Application
A modern, pastel-themed web dashboard for analyzing historical climate data,
visualizing trends, comparing city weather patterns, predicting temperature trends,
and fetching real-time weather observations.

Author: Academic Project Presentation (BTech CSE)
Tech Stack: Python, Streamlit, Pandas, NumPy, Seaborn, Matplotlib, Plotly, Scikit-learn, Requests
"""

import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Import custom utility modules
from utils.data_processing import (
    detect_and_map_columns,
    clean_and_transform_data,
    calculate_summary_metrics,
    generate_automated_narrative,
    COLUMN_ALIASES
)
from utils.visualizations import (
    PASTEL_COLORS,
    plot_daily_temperature_trend,
    plot_monthly_avg_temperature,
    plot_min_max_temp_range,
    plot_daily_rainfall,
    plot_monthly_rainfall,
    plot_humidity_wind_trends,
    plot_temp_vs_humidity_scatter,
    plot_rainfall_vs_humidity_scatter,
    plot_city_comparison_bar,
    plot_seaborn_temp_distribution,
    plot_seaborn_monthly_boxplot,
    plot_seaborn_correlation_heatmap
)
from utils.weather_api import fetch_current_weather
from utils.ml_prediction import train_temperature_trend_model, plot_temperature_prediction

# ---------------------------------------------------------
# Page Configuration & Pastel Custom CSS Injection
# ---------------------------------------------------------
st.set_page_config(
    page_title="Weather Pattern Visualizer",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_pastel_css():
    st.markdown(
        f"""
        <style>
        /* Global Background and Typography */
        .stApp {{
            background-color: {PASTEL_COLORS['background']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {PASTEL_COLORS['text']};
        }}
        
        /* Main Title Header Styling */
        .main-header {{
            background: linear-gradient(135deg, #FFFFFF 0%, {PASTEL_COLORS['blue_light']} 100%);
            padding: 24px 30px;
            border-radius: 16px;
            border: 1px solid {PASTEL_COLORS['border']};
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            margin-bottom: 24px;
        }}
        .main-header h1 {{
            font-size: 30px;
            font-weight: 700;
            color: {PASTEL_COLORS['text']};
            margin: 0 0 6px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .main-header p {{
            font-size: 15px;
            color: {PASTEL_COLORS['text_secondary']};
            margin: 0;
        }}
        
        /* Pastel Card Wrapper */
        .pastel-card {{
            background-color: {PASTEL_COLORS['card_bg']};
            border-radius: 12px;
            padding: 20px;
            border: 1px solid {PASTEL_COLORS['border']};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
            margin-bottom: 20px;
        }}
        
        /* Metric Box Styling */
        .metric-card {{
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 16px 18px;
            border: 1px solid {PASTEL_COLORS['border']};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            border-top: 4px solid {PASTEL_COLORS['blue_main']};
            transition: transform 0.2s ease;
        }}
        .metric-card.lavender {{ border-top-color: {PASTEL_COLORS['lavender_main']}; }}
        .metric-card.mint {{ border-top-color: {PASTEL_COLORS['mint_main']}; }}
        .metric-card.peach {{ border-top-color: {PASTEL_COLORS['peach_main']}; }}
        .metric-card.yellow {{ border-top-color: {PASTEL_COLORS['yellow_main']}; }}
        
        .metric-title {{
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {PASTEL_COLORS['text_secondary']};
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: 700;
            color: {PASTEL_COLORS['text']};
        }}
        .metric-sub {{
            font-size: 12px;
            color: {PASTEL_COLORS['text_secondary']};
            margin-top: 4px;
        }}
        
        /* Subheading Font Sizing */
        .section-heading {{
            font-size: 22px;
            font-weight: 600;
            color: {PASTEL_COLORS['text']};
            margin-top: 10px;
            margin-bottom: 16px;
        }}

        /* Narrative Explanation Box */
        .narrative-box {{
            background-color: {PASTEL_COLORS['lavender_light']};
            border-left: 4px solid {PASTEL_COLORS['lavender_main']};
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.6;
            color: {PASTEL_COLORS['text']};
            margin-top: 15px;
            margin-bottom: 20px;
        }}
        .narrative-box.mint {{
            background-color: {PASTEL_COLORS['mint_light']};
            border-left-color: {PASTEL_COLORS['mint_main']};
        }}
        .narrative-box.blue {{
            background-color: {PASTEL_COLORS['blue_light']};
            border-left-color: {PASTEL_COLORS['blue_main']};
        }}
        .narrative-box.peach {{
            background-color: {PASTEL_COLORS['peach_light']};
            border-left-color: {PASTEL_COLORS['peach_main']};
        }}
        
        /* Badge Pill */
        .badge-pill {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background-color: {PASTEL_COLORS['blue_light']};
            color: {PASTEL_COLORS['text']};
            border: 1px solid {PASTEL_COLORS['border']};
        }}
        
        /* Streamlit Native Overrides */
        div[data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid {PASTEL_COLORS['border']};
        }}
        .stButton>button {{
            background-color: {PASTEL_COLORS['blue_light']};
            color: {PASTEL_COLORS['text']};
            border-radius: 8px;
            border: 1px solid {PASTEL_COLORS['border']};
            font-weight: 600;
            padding: 8px 18px;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background-color: {PASTEL_COLORS['blue_main']};
            color: #FFFFFF;
            border-color: {PASTEL_COLORS['blue_main']};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_pastel_css()

# ---------------------------------------------------------
# Helper Functions for Data Loading
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sample_dataset(file_path: str) -> pd.DataFrame:
    """Loads bundled sample dataset from data directory."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Sample dataset file missing at `{file_path}`.")
        return pd.DataFrame()

# ---------------------------------------------------------
# Sidebar Setup & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 40px; margin-bottom: 5px;">🌦️</div>
            <div style="font-size: 20px; font-weight: 700; color: {PASTEL_COLORS['text']};">Weather Visualizer</div>
            <div style="font-size: 12px; color: {PASTEL_COLORS['text_secondary']};">Academic CSE Presentation</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 📌 Navigation")
    nav_option = st.radio(
        label="Select Section",
        options=[
            "Overview",
            "Historical Analysis",
            "Temperature Trends",
            "Rainfall Analysis",
            "Humidity & Wind Analysis",
            "Weather Comparison",
            "Data Explorer",
            "Live Current Weather (API)",
            "Temperature Prediction (AI)",
            "About Project"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings & Controls")
    
    # 1. Dataset Selection Choice
    dataset_choice = st.selectbox(
        "Data Source",
        options=["Sample Dataset (Single City)", "Sample Dataset (Multi-City)", "Upload Custom CSV"],
        index=0
    )
    
    # Unit Selector
    temp_unit = st.radio("Temperature Unit", options=["°C (Celsius)", "°F (Fahrenheit)"], horizontal=True)
    temp_unit_code = "F" if "Fahrenheit" in temp_unit else "C"

# ---------------------------------------------------------
# Load Active Dataset
# ---------------------------------------------------------
raw_df = pd.DataFrame()
is_custom_upload = False

if dataset_choice == "Sample Dataset (Single City)":
    raw_df = load_sample_dataset("data/sample_weather.csv")
elif dataset_choice == "Sample Dataset (Multi-City)":
    raw_df = load_sample_dataset("data/sample_multi_city.csv")
else:
    is_custom_upload = True
    uploaded_file = st.sidebar.file_uploader("Upload Historical Weather CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        st.info("👈 Upload a CSV file in the sidebar to analyze your weather data.")

# Process dataset if available
df = pd.DataFrame()
cleaning_warnings = []
column_mapping = {}

if not raw_df.empty:
    column_mapping = detect_and_map_columns(raw_df)
    
    # Allow manual mapping overrides in sidebar expander
    with st.sidebar.expander("🛠️ Column Mapping & Config", expanded=is_custom_upload):
        st.caption("Map standard weather features to your dataset columns:")
        mapped_dict = {}
        for std_key in COLUMN_ALIASES.keys():
            options = ["(None)"] + list(raw_df.columns)
            default_idx = 0
            if column_mapping[std_key] in raw_df.columns:
                default_idx = options.index(column_mapping[std_key])
            sel = st.selectbox(f"Map '{std_key}'", options=options, index=default_idx, key=f"map_{std_key}")
            mapped_dict[std_key] = None if sel == "(None)" else sel
        column_mapping = mapped_dict

    try:
        df, cleaning_warnings = clean_and_transform_data(raw_df, column_mapping, temp_unit=temp_unit_code)
    except Exception as e:
        st.error(f"Data Processing Error: {str(e)}")

# Sidebar Date & City Filters
selected_city = None
if not df.empty:
    with st.sidebar:
        # City Filter
        if "city" in df.columns and df["city"].nunique() > 1:
            city_options = ["All Cities"] + list(df["city"].unique())
            city_sel = st.selectbox("Select City", options=city_options, index=0)
            if city_sel != "All Cities":
                selected_city = city_sel
                df = df[df["city"] == selected_city].copy()
                
        # Date Filter
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        
        date_range = st.date_input(
            "Filter Date Range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()

# OpenWeatherMap API Key in Sidebar
with st.sidebar.expander("🔑 OpenWeatherMap API Key", expanded=False):
    st.caption("Optional for live weather observations.")
    user_api_key = st.text_input("Enter API Key", type="password", help="Free API key from openweathermap.org")

# ---------------------------------------------------------
# App Header Render
# ---------------------------------------------------------
current_date_str = datetime.date.today().strftime("%B %d, %Y")
source_badge = "Sample Dataset (Seattle)" if dataset_choice == "Sample Dataset (Single City)" else ("Sample Dataset (Multi-City)" if dataset_choice == "Sample Dataset (Multi-City)" else "Custom Uploaded Dataset")

st.markdown(
    f"""
    <div class="main-header">
        <h1>🌤️ Weather Pattern Visualizer</h1>
        <p>Explore historical weather data, discover patterns, and understand climate trends.</p>
        <div style="margin-top: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span class="badge-pill">📅 Today: {current_date_str}</span>
            <span class="badge-pill" style="background-color: {PASTEL_COLORS['lavender_light']};">📂 Active Source: {source_badge}</span>
            <span class="badge-pill" style="background-color: {PASTEL_COLORS['mint_light']};">🌡️ Unit: {temp_unit_code}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Section 1: Overview Dashboard
# ---------------------------------------------------------
if nav_option == "Overview":
    st.markdown('<div class="section-heading">📊 Overview Dashboard</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning("⚠️ No data available. Please select a valid dataset or upload a CSV file.")
    else:
        metrics = calculate_summary_metrics(df, selected_city=selected_city, temp_unit=temp_unit_code)
        
        # Metric Cards Layout
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        unit = metrics["unit_symbol"]
        
        with c1:
            val = f"{metrics['avg_temp']}{unit}" if metrics['avg_temp'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Temp</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">Mean observed</div>
                </div>
                """, unsafe_allow_html=True
            )
            
        with c2:
            val = f"{metrics['max_temp']}{unit}" if metrics['max_temp'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-card peach">
                    <div class="metric-title">Max Temp</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">Peak recorded</div>
                </div>
                """, unsafe_allow_html=True
            )

        with c3:
            val = f"{metrics['min_temp']}{unit}" if metrics['min_temp'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-card lavender">
                    <div class="metric-title">Min Temp</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">Lowest recorded</div>
                </div>
                """, unsafe_allow_html=True
            )

        with c4:
            val = f"{metrics['total_rainfall']} mm" if metrics['total_rainfall'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-card mint">
                    <div class="metric-title">Total Rain</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{metrics['rainy_days'] or 0} rainy days</div>
                </div>
                """, unsafe_allow_html=True
            )

        with c5:
            val = f"{metrics['avg_humidity']}%" if metrics['avg_humidity'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-card yellow">
                    <div class="metric-title">Avg Humidity</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">Relative humidity</div>
                </div>
                """, unsafe_allow_html=True
            )

        with c6:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Records</div>
                    <div class="metric-value">{metrics['total_records']}</div>
                    <div class="metric-sub">{metrics['date_start']} to {metrics['date_end']}</div>
                </div>
                """, unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main Visual Highlights
        col_left, col_right = st.columns([7, 5])
        
        with col_left:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig_trend = plot_daily_temperature_trend(df, temp_unit=temp_unit_code)
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            if "rainfall" in df.columns and df["rainfall"].notna().any():
                fig_rain = plot_monthly_rainfall(df)
                st.plotly_chart(fig_rain, use_container_width=True)
            else:
                fig_m_temp = plot_monthly_avg_temperature(df, temp_unit=temp_unit_code)
                st.plotly_chart(fig_m_temp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Automated Insights Banner
        narratives = generate_automated_narrative(df, temp_unit=temp_unit_code)
        st.markdown(f'<div class="narrative-box blue"><b>💡 Executive Insights:</b><br>{narratives["temp_narrative"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 2: Historical Analysis
# ---------------------------------------------------------
elif nav_option == "Historical Analysis":
    st.markdown('<div class="section-heading">📂 Historical Dataset Analysis & Quality Check</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.info("Please upload a CSV dataset or select a sample dataset from the sidebar.")
    else:
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        st.subheader("Data Quality & Standard Mapping Status")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Detected Column Mapping:**")
            mapping_df = pd.DataFrame([
                {"Standard Field": k, "Dataset Column": v if v else "⚠️ Missing"} 
                for k, v in column_mapping.items()
            ])
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
            
        with col_b:
            st.write("**Data Hygiene Summary:**")
            st.write(f"- Total Rows: `{len(df)}`")
            st.write(f"- Date Range: `{df['date'].min().strftime('%Y-%m-%d')}` to `{df['date'].max().strftime('%Y-%m-%d')}`")
            st.write(f"- Missing Temperature Values: `{df['temperature'].isna().sum() if 'temperature' in df.columns else 'N/A'}`")
            st.write(f"- Missing Rainfall Values: `{df['rainfall'].isna().sum() if 'rainfall' in df.columns else 'N/A'}`")
            
            if cleaning_warnings:
                st.warning("⚠️ Data Cleaning Notes:\n" + "\n".join([f"- {w}" for w in cleaning_warnings]))
            else:
                st.success("✅ Dataset is clean with zero missing dates or corrupted rows.")
                
        st.markdown('</div>', unsafe_allow_html=True)

        # Dataset Preview Table
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        st.subheader("Dataset Preview (First 15 Rows)")
        st.dataframe(df.head(15), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 3: Temperature Trends
# ---------------------------------------------------------
elif nav_option == "Temperature Trends":
    st.markdown('<div class="section-heading">🌡️ Temperature Variation & Seasonal Trends</div>', unsafe_allow_html=True)
    
    if df.empty or "temperature" not in df.columns:
        st.warning("Temperature data is not available in the current dataset.")
    else:
        # Chart 1: Daily Variation
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        fig1 = plot_daily_temperature_trend(df, temp_unit=temp_unit_code)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Charts 2 & 3: Monthly Avg & Min/Max Range
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig2 = plot_monthly_avg_temperature(df, temp_unit=temp_unit_code)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_right:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig3 = plot_min_max_temp_range(df, temp_unit=temp_unit_code)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Seaborn Statistical Plots
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        st.subheader("Statistical Distributions (Seaborn & Matplotlib)")
        s_left, s_right = st.columns(2)
        
        with s_left:
            fig_seaborn_dist = plot_seaborn_temp_distribution(df, temp_unit=temp_unit_code)
            if fig_seaborn_dist:
                st.pyplot(fig_seaborn_dist)
                
        with s_right:
            fig_seaborn_box = plot_seaborn_monthly_boxplot(df, temp_unit=temp_unit_code)
            if fig_seaborn_box:
                st.pyplot(fig_seaborn_box)
                
        st.markdown('</div>', unsafe_allow_html=True)

        # Automated Trend Explanation
        narratives = generate_automated_narrative(df, temp_unit=temp_unit_code)
        st.markdown(f'<div class="narrative-box peach"><b>📊 Temperature Trend Explanation:</b><br>{narratives["temp_narrative"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 4: Rainfall Analysis
# ---------------------------------------------------------
elif nav_option == "Rainfall Analysis":
    st.markdown('<div class="section-heading">🌧️ Precipitation & Rainfall Analysis</div>', unsafe_allow_html=True)
    
    if df.empty or "rainfall" not in df.columns or not df["rainfall"].notna().any():
        st.warning("⚠️ Rainfall information is unavailable in the current dataset. Please upload a dataset containing a precipitation or rainfall column (e.g., 'rainfall', 'precipitation', 'precip', 'rain').")
    else:
        # Metrics Row
        r1, r2, r3 = st.columns(3)
        total_r = round(df["rainfall"].sum(), 1)
        rainy_d = int((df["rainfall"] > 0.1).sum())
        wettest_day_val = round(df["rainfall"].max(), 1)
        
        with r1:
            st.markdown(f'<div class="metric-card mint"><div class="metric-title">Total Precipitation</div><div class="metric-value">{total_r} mm</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Rainy Days</div><div class="metric-value">{rainy_d} Days</div><div class="metric-sub">{(rainy_d/len(df))*100:.1f}% of total days</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card peach"><div class="metric-title">Max Single-Day Rain</div><div class="metric-value">{wettest_day_val} mm</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Rainfall Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig_daily_r = plot_daily_rainfall(df)
            if fig_daily_r:
                st.plotly_chart(fig_daily_r, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig_monthly_r = plot_monthly_rainfall(df)
            if fig_monthly_r:
                st.plotly_chart(fig_monthly_r, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Explanation
        narratives = generate_automated_narrative(df, temp_unit=temp_unit_code)
        st.markdown(f'<div class="narrative-box mint"><b>🌧️ Precipitation Pattern Summary:</b><br>{narratives["rain_narrative"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 5: Humidity & Wind Analysis
# ---------------------------------------------------------
elif nav_option == "Humidity & Wind Analysis":
    st.markdown('<div class="section-heading">💨 Humidity & Wind Speed Dynamics</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available for humidity and wind analysis.")
    else:
        # Time Series
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        fig_hw = plot_humidity_wind_trends(df)
        st.plotly_chart(fig_hw, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Correlation Scatter Plots
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig_scat1 = plot_temp_vs_humidity_scatter(df, temp_unit=temp_unit_code)
            if fig_scat1:
                st.plotly_chart(fig_scat1, use_container_width=True)
            else:
                st.info("Temperature or Humidity column missing for correlation plot.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig_scat2 = plot_rainfall_vs_humidity_scatter(df)
            if fig_scat2:
                st.plotly_chart(fig_scat2, use_container_width=True)
            else:
                st.info("Rainfall data missing for rainfall-humidity correlation plot.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Heatmap
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        st.subheader("Multivariate Feature Correlation Heatmap")
        fig_hm = plot_seaborn_correlation_heatmap(df)
        if fig_hm:
            st.pyplot(fig_hm)
        st.markdown('</div>', unsafe_allow_html=True)

        narratives = generate_automated_narrative(df, temp_unit=temp_unit_code)
        st.markdown(f'<div class="narrative-box blue"><b>💡 Humidity & Wind Insights:</b><br>{narratives["humidity_narrative"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 6: Weather Comparison
# ---------------------------------------------------------
elif nav_option == "Weather Comparison":
    st.markdown('<div class="section-heading">🏙️ Multi-City Weather Comparison</div>', unsafe_allow_html=True)
    
    # Check if multiple cities are present
    if df.empty:
        st.warning("No dataset loaded.")
    elif "city" not in df.columns or df["city"].nunique() <= 1:
        st.markdown(
            f"""
            <div class="narrative-box lavender">
                📌 <b>Single City Dataset Detected:</b><br>
                The active dataset currently contains data for only one city 
                (<b>{df['city'].iloc[0] if 'city' in df.columns else 'Selected Location'}</b>).<br><br>
                To explore multi-city comparison charts, click the button below to switch to the <b>Multi-City Sample Dataset</b>!
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🔄 Load Multi-City Sample Dataset"):
            st.sidebar.selectbox("Data Source", options=["Sample Dataset (Multi-City)"], index=0)
            st.rerun()
    else:
        st.info(f"Comparing weather observations across **{df['city'].nunique()} cities**: {', '.join(df['city'].unique())}")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig_c_temp = plot_city_comparison_bar(df, "temperature", "Average Temperature", temp_unit=temp_unit_code)
            if fig_c_temp:
                st.plotly_chart(fig_c_temp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with comp_col2:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            fig_c_rain = plot_city_comparison_bar(df, "rainfall", "Total Precipitation", temp_unit=temp_unit_code)
            if fig_c_rain:
                st.plotly_chart(fig_c_rain, use_container_width=True)
            else:
                fig_c_hum = plot_city_comparison_bar(df, "humidity", "Average Humidity", temp_unit=temp_unit_code)
                if fig_c_hum:
                    st.plotly_chart(fig_c_hum, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Detailed Comparison Table
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        st.subheader("Statistical City Comparison Summary Table")
        
        agg_dict = {}
        if "temperature" in df.columns:
            agg_dict["temperature"] = ["mean", "min", "max"]
        if "rainfall" in df.columns:
            agg_dict["rainfall"] = ["sum", "mean"]
        if "humidity" in df.columns:
            agg_dict["humidity"] = ["mean"]
        if "wind_speed" in df.columns:
            agg_dict["wind_speed"] = ["mean"]
            
        summary_table = df.groupby("city").agg(agg_dict)
        summary_table.columns = ['_'.join(c).strip() for c in summary_table.columns]
        summary_table = summary_table.round(1)
        st.dataframe(summary_table, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 7: Data Explorer
# ---------------------------------------------------------
elif nav_option == "Data Explorer":
    st.markdown('<div class="section-heading">🔍 Data Explorer & Export</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available.")
    else:
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        
        # Explorer Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            search_query = st.text_input("🔍 Search Date or City", placeholder="e.g. 2024-06 or Seattle")
        with f2:
            selected_cols = st.multiselect("Select Display Columns", options=list(df.columns), default=list(df.columns)[:7])
        with f3:
            st.write("&nbsp;")
            # CSV Download Button
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name=f"cleaned_weather_data_{datetime.date.today()}.csv",
                mime="text/csv"
            )
            
        # Filtering logic
        filtered_df = df.copy()
        if search_query:
            mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            filtered_df = filtered_df[mask]
            
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
        
        if selected_cols:
            st.dataframe(filtered_df[selected_cols], use_container_width=True)
        else:
            st.dataframe(filtered_df, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

        # Descriptive Statistics
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        st.subheader("Descriptive Statistics Summary (`df.describe()`)")
        st.dataframe(df.describe().round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 8: Live Current Weather (OpenWeatherMap API)
# ---------------------------------------------------------
elif nav_option == "Live Current Weather (API)":
    st.markdown('<div class="section-heading">🌤️ Live Current Weather API Integration</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="narrative-box blue">
            <b>ℹ️ Real-Time Weather Observation Module:</b><br>
            Connects to the OpenWeatherMap REST API to retrieve current weather conditions. 
            <i>Note: Live weather observations are retrieved directly from satellite/station APIs and are distinct from historical datasets.</i>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_input, col_unit = st.columns([3, 1])
    with col_input:
        city_input = st.text_input("Enter City Name", value="London", placeholder="e.g. New York, Tokyo, New Delhi")
    with col_unit:
        unit_choice = st.selectbox("API Units", options=["Metric (°C, m/s)", "Imperial (°F, mph)"])
        api_units = "imperial" if "Imperial" in unit_choice else "metric"

    if st.button("🔍 Fetch Live Weather"):
        api_key_to_use = user_api_key if user_api_key else None
        
        with st.spinner(f"Fetching real-time weather observations for {city_input}..."):
            result = fetch_current_weather(city_input, api_key=api_key_to_use, units=api_units)
            
        if result["success"]:
            st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
            
            c_header, c_icon = st.columns([4, 1])
            with c_header:
                st.markdown(f"### {result['city']}, {result['country']}")
                st.caption(f"Observed at {result['dt_timestamp']}")
                st.markdown(f"#### **{result['description']}**")
            with c_icon:
                st.image(result["icon_url"], width=90)
                
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Temperature</div><div class="metric-value">{result["temp"]}{result["units"]}</div><div class="metric-sub">Feels like {result["feels_like"]}{result["units"]}</div></div>', unsafe_allow_html=True)
            with w2:
                st.markdown(f'<div class="metric-card peach"><div class="metric-title">Temp Range</div><div class="metric-value">{result["temp_min"]} / {result["temp_max"]}</div><div class="metric-sub">Min / Max {result["units"]}</div></div>', unsafe_allow_html=True)
            with w3:
                st.markdown(f'<div class="metric-card mint"><div class="metric-title">Humidity</div><div class="metric-value">{result["humidity"]}%</div><div class="metric-sub">Pressure {result["pressure"]} hPa</div></div>', unsafe_allow_html=True)
            with w4:
                st.markdown(f'<div class="metric-card lavender"><div class="metric-title">Wind Speed</div><div class="metric-value">{result["wind_speed"]} {result["speed_units"]}</div></div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if result["error_type"] == "MISSING_KEY":
                st.warning(f"⚠️ {result['message']}")
                st.markdown(
                    """
                    <div class="pastel-card">
                        <h4>🔑 How to configure your free OpenWeatherMap API Key:</h4>
                        <ol>
                            <li>Sign up for a free API key at <a href="https://openweathermap.org/api" target="_blank">openweathermap.org/api</a>.</li>
                            <li>Enter the key in the sidebar expander <b>"🔑 OpenWeatherMap API Key"</b>.</li>
                            <li>Or add <code>OPENWEATHER_API_KEY = "your_key"</code> to <code>.streamlit/secrets.toml</code> for local development / Render environment variables.</li>
                        </ol>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error(f"❌ API Error: {result['message']}")

# ---------------------------------------------------------
# Section 9: Temperature Trend Prediction (AI/ML)
# ---------------------------------------------------------
elif nav_option == "Temperature Prediction (AI)":
    st.markdown('<div class="section-heading">📈 Temperature Trend Prediction (Scikit-Learn ML)</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="narrative-box peach">
            <b>⚠️ Educational Disclaimer:</b><br>
            This section uses Scikit-learn (Linear / Polynomial Regression) to fit a statistical trendline over historical dates and project future trend directions. 
            This represents an <b>educational statistical estimate</b> for demonstration purposes, not an operational physics-based weather forecast model.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if df.empty or "temperature" not in df.columns:
        st.warning("Historical temperature data is required to train the regression model.")
    elif len(df.dropna(subset=["temperature"])) < 15:
        st.warning("Minimum 15 historical temperature records required to fit regression model.")
    else:
        st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
        
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            forecast_horizon = st.slider("Forecast Horizon (Days into Future)", min_value=15, max_value=90, value=30, step=15)
        with m_c2:
            model_mode = st.radio("Model Architecture", options=["Polynomial Regression (Degree 2)", "Linear Regression"], horizontal=True)
            use_poly = "Polynomial" in model_mode
            
        metrics, pred_df = train_temperature_trend_model(df, forecast_days=forecast_horizon, use_polynomial=use_poly, temp_unit=temp_unit_code)
        
        # Display Model Performance Metrics
        pm1, pm2, pm3, pm4 = st.columns(4)
        with pm1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Model R² Score</div><div class="metric-value">{metrics["r2_score"]}</div><div class="metric-sub">Goodness of fit</div></div>', unsafe_allow_html=True)
        with pm2:
            st.markdown(f'<div class="metric-card lavender"><div class="metric-title">Mean Abs Error (MAE)</div><div class="metric-value">{metrics["mae"]}°</div><div class="metric-sub">Avg deviation</div></div>', unsafe_allow_html=True)
        with pm3:
            st.markdown(f'<div class="metric-card mint"><div class="metric-title">Annual Trend Rate</div><div class="metric-value">{metrics["trend_rate"]}</div><div class="metric-sub">Linear slope</div></div>', unsafe_allow_html=True)
        with pm4:
            st.markdown(f'<div class="metric-card yellow"><div class="metric-title">Training Samples</div><div class="metric-value">{metrics["total_training_samples"]}</div><div class="metric-sub">Historical points</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        fig_pred = plot_temperature_prediction(pred_df, temp_unit=temp_unit_code)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 10: About Project
# ---------------------------------------------------------
elif nav_option == "About Project":
    st.markdown('<div class="section-heading">ℹ️ About Project & Academic Viva Notes</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="pastel-card">', unsafe_allow_html=True)
    st.markdown(
        """
        ### 🎯 Project Overview
        **Weather Pattern Visualizer** is a full-stack, data science web application designed to analyze historical meteorological records, detect long-term climate patterns, visualize multivariate relationships, and provide statistical forecasting models.
        
        Developed as a **BTech Computer Science & Engineering (CSE)** minor project demonstration.

        ---

        ### 🧠 Problem Statement & Objectives
        - **Problem:** Raw meteorological datasets are often messy, high-dimensional, and difficult for non-experts to interpret without interactive visual tools.
        - **Objective:** Build an intuitive, beginner-friendly visual dashboard with modern soft pastel UI aesthetics to auto-clean weather CSVs, compute statistical summaries, detect anomalies, compare multi-city microclimates, and integrate real-time REST API observations.

        ---

        ### 🛠️ Technology Stack
        - **Core Language:** Python 3.11+
        - **Web Framework:** Streamlit
        - **Data Manipulation & Analysis:** Pandas, NumPy
        - **Interactive Plotting:** Plotly (Plotly Express & Graph Objects)
        - **Statistical Graphics:** Seaborn, Matplotlib
        - **Machine Learning:** Scikit-Learn (Linear & Polynomial Regression)
        - **API Client:** Requests (OpenWeatherMap REST API)
        - **Deployment Target:** Render (Web Service Platform)

        ---

        ### ✨ Key Features Implemented
        1. **Fuzzy Column Auto-Detection:** Automatically maps unknown CSV column headers to standardized meteorological variables.
        2. **Dynamic Date & Location Filtering:** Interactively updates metrics and charts across user-selected date ranges and cities.
        3. **Multi-City Weather Comparison:** Side-by-side statistical comparison across multiple geographical locations.
        4. **Seaborn & Matplotlib Integration:** Statistical Kernel Density Estimations (KDE), Monthly Boxplots, and Correlation Heatmaps.
        5. **Automated Data Insights:** Rule-based natural language narrative generation for executive summaries.
        6. **Scikit-Learn Machine Learning:** Statistical trendline fitting and short-term temperature projection with evaluation metrics ($R^2$, MAE, RMSE).
        7. **OpenWeatherMap API Integration:** Secure, real-time live weather lookup with error handling.

        ---

        ### 🎓 Academic Viva Questions & Answers
        1. **Q: How does the application handle missing CSV columns or invalid dates?**  
           *A: The system uses pandas coerce logic to convert dates safely, drops rows with invalid timestamps, fills missing derived fields where possible, and provides graceful empty-state warnings without crashing.*
           
        2. **Q: Why use Plotly alongside Seaborn?**  
           *A: Plotly provides client-side interactive zooming, panning, and tooltips for time-series exploration, whereas Seaborn generates static publication-quality statistical distributions (KDE, boxplots, heatmaps).*
           
        3. **Q: How are API keys protected in production?**  
           *A: Keys are accessed via Streamlit Secrets (`st.secrets`) or environment variables, preventing hardcoded credentials from entering version control.*
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
