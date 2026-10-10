"""
OpenWeatherMap API Integration Module for Weather Pattern Visualizer.
Handles fetching live weather observation data, parsing JSON responses,
and managing errors gracefully.
"""

import os
import requests
from typing import Dict, Any, Optional

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_current_weather(city_name: str, api_key: Optional[str] = None, units: str = "metric") -> Dict[str, Any]:
    """
    Fetches current weather observations for a given city from OpenWeatherMap API.
    
    Parameters:
    - city_name: City name string (e.g. 'London', 'New York', 'New Delhi')
    - api_key: Optional OpenWeatherMap API key override
    - units: 'metric' for Celsius/m/s, 'imperial' for Fahrenheit/mph
    
    Returns:
    - Dictionary with status, parsed weather metrics, icon URL, or error details.
    """
    # Try retrieving key from argument -> environment -> streamlit secrets safely
    key = api_key or os.environ.get("OPENWEATHER_API_KEY")
    
    if not key:
        try:
            import streamlit as st
            if "OPENWEATHER_API_KEY" in st.secrets:
                key = st.secrets["OPENWEATHER_API_KEY"]
        except Exception:
            pass

    if not key or key == "your_openweather_api_key_here":
        return {
            "success": False,
            "error_type": "MISSING_KEY",
            "message": "OpenWeatherMap API Key is not configured. Historical analysis remains fully functional!"
        }

    if not city_name or not city_name.strip():
        return {
            "success": False,
            "error_type": "INVALID_CITY",
            "message": "Please enter a valid city name."
        }

    try:
        params = {
            "q": city_name.strip(),
            "appid": key,
            "units": units
        }
        
        response = requests.get(BASE_URL, params=params, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            
            # Unit labels
            temp_unit = "°F" if units == "imperial" else "°C"
            speed_unit = "mph" if units == "imperial" else "m/s"
            
            icon_code = data["weather"][0]["icon"] if data.get("weather") else "01d"
            
            return {
                "success": True,
                "city": data.get("name", city_name),
                "country": data.get("sys", {}).get("country", ""),
                "temp": round(data["main"]["temp"], 1),
                "feels_like": round(data["main"]["feels_like"], 1),
                "temp_min": round(data["main"]["temp_min"], 1),
                "temp_max": round(data["main"]["temp_max"], 1),
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": round(data["wind"]["speed"], 1),
                "description": data["weather"][0]["description"].title() if data.get("weather") else "Clear",
                "icon_url": f"https://openweathermap.org/img/wn/{icon_code}@2x.png",
                "units": temp_unit,
                "speed_units": speed_unit,
                "dt_timestamp": pd_dt_now_str(data.get("dt"))
            }
        elif response.status_code == 404:
            return {
                "success": False,
                "error_type": "CITY_NOT_FOUND",
                "message": f"City '{city_name}' was not found. Please check the spelling."
            }
        elif response.status_code == 401:
            return {
                "success": False,
                "error_type": "UNAUTHORIZED",
                "message": "Invalid API key provided. Please verify your OpenWeatherMap API key."
            }
        elif response.status_code == 429:
            return {
                "success": False,
                "error_type": "RATE_LIMIT",
                "message": "OpenWeatherMap API rate limit exceeded. Please try again later."
            }
        else:
            return {
                "success": False,
                "error_type": "HTTP_ERROR",
                "message": f"API request returned HTTP status code {response.status_code}."
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_type": "TIMEOUT",
            "message": "Connection to OpenWeatherMap timed out. Check your internet connection."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error_type": "CONNECTION_ERROR",
            "message": "Failed to connect to weather service network."
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "UNKNOWN",
            "message": f"An unexpected error occurred: {str(e)}"
        }


def pd_dt_now_str(timestamp=None):
    from datetime import datetime
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
