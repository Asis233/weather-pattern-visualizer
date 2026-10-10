"""
Visualization Utility Module for Weather Pattern Visualizer.
Provides Plotly interactive charts and Seaborn/Matplotlib statistical visualizations 
adhering to a soft, modern pastel theme.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as px_go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any

# Pastel Color Palette Definitions
PASTEL_COLORS = {
    "background": "#F8F7FC",
    "card_bg": "#FFFFFF",
    "text": "#343447",
    "text_secondary": "#73738A",
    "border": "#E7E5EF",
    "blue_main": "#5C90D2",
    "blue_light": "#C9E4F6",
    "lavender_main": "#9B82C9",
    "lavender_light": "#E5DDF8",
    "mint_main": "#48B188",
    "mint_light": "#D9F2E6",
    "peach_main": "#E88564",
    "peach_light": "#FFE3D5",
    "yellow_main": "#E5C158",
    "yellow_light": "#FFF2C6",
}

# Standardized Plotly Layout Settings
def apply_plotly_theme(fig, title: str = "", x_title: str = "", y_title: str = ""):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color=PASTEL_COLORS["text"], family="Inter, sans-serif"),
            x=0.01,
            xanchor="left"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=30, t=50, b=40),
        font=dict(family="Inter, sans-serif", color=PASTEL_COLORS["text"], size=13),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=13, color=PASTEL_COLORS["text_secondary"])),
            showgrid=True,
            gridcolor=PASTEL_COLORS["border"],
            zeroline=False,
            showline=True,
            linecolor=PASTEL_COLORS["border"]
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=13, color=PASTEL_COLORS["text_secondary"])),
            showgrid=True,
            gridcolor=PASTEL_COLORS["border"],
            zeroline=False,
            showline=True,
            linecolor=PASTEL_COLORS["border"]
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor=PASTEL_COLORS["border"]
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=PASTEL_COLORS["border"],
            borderwidth=1,
            font=dict(size=12, color=PASTEL_COLORS["text"])
        ),
        autosize=True
    )
    return fig


# ---------------------------------------------------------
# 1. Temperature Visualizations
# ---------------------------------------------------------

def plot_daily_temperature_trend(df: pd.DataFrame, temp_unit: str = "C"):
    """Interactive daily temperature time-series with area fill."""
    unit = "°F" if temp_unit == "F" else "°C"
    
    fig = px_go.Figure()
    
    if "temperature" in df.columns:
        fig.add_trace(px_go.Scatter(
            x=df["date"],
            y=df["temperature"],
            mode="lines",
            name=f"Avg Temp ({unit})",
            line=dict(color=PASTEL_COLORS["blue_main"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(201, 228, 246, 0.4)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Temp:</b> %{y:.1f}" + unit + "<extra></extra>"
        ))
        
    # Optional 7-day rolling average
    if len(df) >= 7 and "temperature" in df.columns:
        df["rolling_temp"] = df["temperature"].rolling(window=7, min_periods=1).mean()
        fig.add_trace(px_go.Scatter(
            x=df["date"],
            y=df["rolling_temp"],
            mode="lines",
            name="7-Day Moving Avg",
            line=dict(color=PASTEL_COLORS["peach_main"], width=2, dash="dash"),
            hovertemplate="<b>7-Day Avg:</b> %{y:.1f}" + unit + "<extra></extra>"
        ))

    apply_plotly_theme(fig, "Daily Temperature Variation", "Date", f"Temperature ({unit})")
    return fig


def plot_monthly_avg_temperature(df: pd.DataFrame, temp_unit: str = "C"):
    """Monthly average temperature bar chart with pastel lavender bars."""
    unit = "°F" if temp_unit == "F" else "°C"
    
    monthly = df.groupby(["month", "month_name"], observed=False)["temperature"].mean().reset_index()
    monthly.sort_values(by="month", inplace=True)
    
    fig = px.bar(
        monthly,
        x="month_name",
        y="temperature",
        text_auto=".1f",
        labels={"month_name": "Month", "temperature": f"Avg Temp ({unit})"},
        color_discrete_sequence=[PASTEL_COLORS["lavender_main"]]
    )
    fig.update_traces(
        marker=dict(cornerradius=6, line=dict(color=PASTEL_COLORS["border"], width=1)),
        textposition="outside"
    )
    apply_plotly_theme(fig, "Average Temperature by Month", "Month", f"Average Temperature ({unit})")
    return fig


def plot_min_max_temp_range(df: pd.DataFrame, temp_unit: str = "C"):
    """Filled temperature range band between Minimum and Maximum daily temps."""
    unit = "°F" if temp_unit == "F" else "°C"
    
    fig = px_go.Figure()
    
    if "temp_max" in df.columns and "temp_min" in df.columns:
        # Max temp line
        fig.add_trace(px_go.Scatter(
            x=df["date"],
            y=df["temp_max"],
            mode="lines",
            name=f"Max Temp ({unit})",
            line=dict(color=PASTEL_COLORS["peach_main"], width=2),
            hovertemplate="<b>Max Temp:</b> %{y:.1f}" + unit + "<extra></extra>"
        ))
        # Min temp line with shaded range fill
        fig.add_trace(px_go.Scatter(
            x=df["date"],
            y=df["temp_min"],
            mode="lines",
            name=f"Min Temp ({unit})",
            line=dict(color=PASTEL_COLORS["blue_main"], width=2),
            fill="tonexty",
            fillcolor="rgba(229, 221, 248, 0.4)",
            hovertemplate="<b>Min Temp:</b> %{y:.1f}" + unit + "<extra></extra>"
        ))
        
    apply_plotly_theme(fig, "Minimum vs Maximum Temperature Range", "Date", f"Temperature ({unit})")
    return fig


# ---------------------------------------------------------
# 2. Rainfall Visualizations
# ---------------------------------------------------------

def plot_daily_rainfall(df: pd.DataFrame):
    """Daily rainfall bar chart in pastel mint green."""
    if "rainfall" not in df.columns or not df["rainfall"].notna().any():
        return None
        
    fig = px.bar(
        df,
        x="date",
        y="rainfall",
        labels={"date": "Date", "rainfall": "Rainfall (mm)"},
        color_discrete_sequence=[PASTEL_COLORS["mint_main"]]
    )
    fig.update_traces(marker=dict(cornerradius=3))
    apply_plotly_theme(fig, "Daily Precipitation / Rainfall", "Date", "Rainfall (mm)")
    return fig


def plot_monthly_rainfall(df: pd.DataFrame):
    """Monthly total rainfall highlighting wettest/driest months."""
    if "rainfall" not in df.columns or not df["rainfall"].notna().any():
        return None

    monthly = df.groupby(["month", "month_name"], observed=False)["rainfall"].sum().reset_index()
    monthly.sort_values(by="month", inplace=True)

    fig = px.bar(
        monthly,
        x="month_name",
        y="rainfall",
        text_auto=".1f",
        labels={"month_name": "Month", "rainfall": "Total Rainfall (mm)"},
        color="rainfall",
        color_continuous_scale=[PASTEL_COLORS["mint_light"], PASTEL_COLORS["mint_main"]]
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(
        marker=dict(cornerradius=6),
        textposition="outside"
    )
    apply_plotly_theme(fig, "Monthly Rainfall Totals", "Month", "Total Rainfall (mm)")
    return fig


# ---------------------------------------------------------
# 3. Humidity & Wind Visualizations
# ---------------------------------------------------------

def plot_humidity_wind_trends(df: pd.DataFrame):
    """Subplots for Humidity and Wind Speed trends."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Relative Humidity (%)", "Wind Speed (km/h)")
    )
    
    if "humidity" in df.columns:
        fig.add_trace(
            px_go.Scatter(
                x=df["date"], y=df["humidity"],
                mode="lines", name="Humidity (%)",
                line=dict(color=PASTEL_COLORS["blue_main"], width=2)
            ),
            row=1, col=1
        )
        
    if "wind_speed" in df.columns:
        fig.add_trace(
            px_go.Scatter(
                x=df["date"], y=df["wind_speed"],
                mode="lines", name="Wind Speed (km/h)",
                line=dict(color=PASTEL_COLORS["lavender_main"], width=2)
            ),
            row=2, col=1
        )

    apply_plotly_theme(fig, "Humidity & Wind Speed Trends", "Date", "Value")
    return fig


def plot_temp_vs_humidity_scatter(df: pd.DataFrame, temp_unit: str = "C"):
    """Scatter plot displaying relationship between Temperature and Relative Humidity."""
    if "temperature" not in df.columns or "humidity" not in df.columns:
        return None

    unit = "°F" if temp_unit == "F" else "°C"
    
    fig = px.scatter(
        df,
        x="temperature",
        y="humidity",
        trendline="ols",
        trendline_color_override=PASTEL_COLORS["peach_main"],
        labels={"temperature": f"Temperature ({unit})", "humidity": "Relative Humidity (%)"},
        color_discrete_sequence=[PASTEL_COLORS["blue_main"]],
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#FFFFFF")))
    apply_plotly_theme(fig, f"Temperature vs Humidity Correlation", f"Temperature ({unit})", "Relative Humidity (%)")
    return fig


def plot_rainfall_vs_humidity_scatter(df: pd.DataFrame):
    """Scatter plot displaying relationship between Rainfall and Relative Humidity."""
    if "rainfall" not in df.columns or "humidity" not in df.columns:
        return None

    fig = px.scatter(
        df[df["rainfall"] > 0],
        x="humidity",
        y="rainfall",
        size="rainfall",
        labels={"humidity": "Relative Humidity (%)", "rainfall": "Rainfall (mm)"},
        color_discrete_sequence=[PASTEL_COLORS["mint_main"]],
        opacity=0.75
    )
    apply_plotly_theme(fig, "Rainfall Intensity vs Relative Humidity (Rainy Days)", "Relative Humidity (%)", "Rainfall (mm)")
    return fig


# ---------------------------------------------------------
# 4. Multi-City Weather Comparison
# ---------------------------------------------------------

def plot_city_comparison_bar(df: pd.DataFrame, metric_col: str, metric_title: str, temp_unit: str = "C"):
    """Grouped bar chart comparing metrics across cities."""
    if "city" not in df.columns:
        return None

    unit = "°F" if temp_unit == "F" else "°C"
    unit_label = unit if "temp" in metric_col else ("mm" if "rain" in metric_col else "%")
    
    if metric_col == "rainfall":
        summary = df.groupby("city")["rainfall"].sum().reset_index()
    else:
        summary = df.groupby("city")[metric_col].mean().reset_index()

    fig = px.bar(
        summary,
        x="city",
        y=metric_col,
        color="city",
        text_auto=".1f",
        labels={"city": "City / Location", metric_col: f"{metric_title} ({unit_label})"},
        color_discrete_sequence=[
            PASTEL_COLORS["blue_main"], 
            PASTEL_COLORS["lavender_main"], 
            PASTEL_COLORS["mint_main"], 
            PASTEL_COLORS["peach_main"]
        ]
    )
    fig.update_traces(
        marker=dict(cornerradius=8, line=dict(color=PASTEL_COLORS["border"], width=1)),
        textposition="outside"
    )
    apply_plotly_theme(fig, f"City Comparison: {metric_title}", "City", f"{metric_title} ({unit_label})")
    return fig


# ---------------------------------------------------------
# 5. Seaborn / Matplotlib Statistical Visualizations
# ---------------------------------------------------------

def plot_seaborn_temp_distribution(df: pd.DataFrame, temp_unit: str = "C"):
    """Clean Seaborn KDE & Distribution plot for temperature."""
    if "temperature" not in df.columns or df["temperature"].dropna().empty:
        return None

    unit = "°F" if temp_unit == "F" else "°C"
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Custom aesthetic configuration
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    sns.histplot(
        df["temperature"].dropna(),
        kde=True,
        ax=ax,
        color=PASTEL_COLORS["blue_main"],
        edgecolor=PASTEL_COLORS["border"],
        linewidth=1,
        alpha=0.4
    )
    
    ax.set_title(f"Temperature Statistical Distribution ({unit})", fontsize=14, fontweight="bold", color=PASTEL_COLORS["text"], pad=15)
    ax.set_xlabel(f"Temperature ({unit})", fontsize=11, color=PASTEL_COLORS["text_secondary"])
    ax.set_ylabel("Frequency / Density", fontsize=11, color=PASTEL_COLORS["text_secondary"])
    
    # Clean spines
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.5, color=PASTEL_COLORS["border"])
    plt.tight_layout()
    return fig


def plot_seaborn_monthly_boxplot(df: pd.DataFrame, temp_unit: str = "C"):
    """Seaborn monthly temperature boxplot showing distribution and spreads."""
    if "temperature" not in df.columns or "month_name" not in df.columns:
        return None

    unit = "°F" if temp_unit == "F" else "°C"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    # Ensure correct month order
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    available_months = [m for m in months_order if m in df["month_name"].unique()]
    
    pastel_12 = [
        PASTEL_COLORS["blue_light"], PASTEL_COLORS["lavender_light"], PASTEL_COLORS["mint_light"], PASTEL_COLORS["peach_light"],
        PASTEL_COLORS["yellow_light"], PASTEL_COLORS["blue_light"], PASTEL_COLORS["lavender_light"], PASTEL_COLORS["mint_light"],
        PASTEL_COLORS["peach_light"], PASTEL_COLORS["yellow_light"], PASTEL_COLORS["blue_light"], PASTEL_COLORS["lavender_light"]
    ]
    
    sns.boxplot(
        data=df,
        x="month_name",
        y="temperature",
        hue="month_name",
        legend=False,
        order=available_months,
        ax=ax,
        palette=pastel_12[:len(available_months)],
        boxprops=dict(edgecolor=PASTEL_COLORS["lavender_main"], linewidth=1.5),
        whiskerprops=dict(color=PASTEL_COLORS["lavender_main"], linewidth=1.5),
        capprops=dict(color=PASTEL_COLORS["lavender_main"], linewidth=1.5),
        medianprops=dict(color=PASTEL_COLORS["peach_main"], linewidth=2)
    )

    ax.set_title(f"Monthly Temperature Distribution & Variances", fontsize=14, fontweight="bold", color=PASTEL_COLORS["text"], pad=15)
    ax.set_xlabel("Month", fontsize=11, color=PASTEL_COLORS["text_secondary"])
    ax.set_ylabel(f"Temperature ({unit})", fontsize=11, color=PASTEL_COLORS["text_secondary"])
    
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.5, color=PASTEL_COLORS["border"])
    plt.tight_layout()
    return fig


def plot_seaborn_correlation_heatmap(df: pd.DataFrame):
    """Seaborn correlation matrix heatmap of numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out calendar IDs
    feature_cols = [c for c in numeric_cols if c not in ["year", "month", "day", "is_rainy"]]
    
    if len(feature_cols) < 2:
        return None

    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=1.5,
        linecolor="#FFFFFF"
    )

    ax.set_title("Weather Factors Correlation Matrix", fontsize=13, fontweight="bold", color=PASTEL_COLORS["text"], pad=15)
    plt.tight_layout()
    return fig
