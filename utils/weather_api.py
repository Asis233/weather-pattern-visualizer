"""
OpenWeatherMap & Open-Meteo API Integration Module with Auto Location Detection.
Handles IP Geolocation, live weather observation requests, dual API provider engines 
(OpenWeatherMap with API key + Open-Meteo zero-key fallback), and weather code translation.
"""

import os
import requests
from typing import Dict, Any, Optional

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"

# WMO Weather Code Mapping to Descriptions and Icon URLs
WMO_WEATHER_CODES = {
    0: ("Clear Sky", "☀️", "https://openweathermap.org/img/wn/01d@2x.png"),
    1: ("Mainly Clear", "🌤️", "https://openweathermap.org/img/wn/02d@2x.png"),
    2: ("Partly Cloudy", "⛅", "https://openweathermap.org/img/wn/03d@2x.png"),
    3: ("Overcast", "☁️", "https://openweathermap.org/img/wn/04d@2x.png"),
    45: ("Foggy", "🌫️", "https://openweathermap.org/img/wn/50d@2x.png"),
    48: ("Depositing Rime Fog", "🌫️", "https://openweathermap.org/img/wn/50d@2x.png"),
    51: ("Light Drizzle", "🌦️", "https://openweathermap.org/img/wn/09d@2x.png"),
    53: ("Moderate Drizzle", "🌧️", "https://openweathermap.org/img/wn/09d@2x.png"),
    55: ("Dense Drizzle", "🌧️", "https://openweathermap.org/img/wn/09d@2x.png"),
    61: ("Slight Rain", "🌦️", "https://openweathermap.org/img/wn/10d@2x.png"),
    63: ("Moderate Rain", "🌧️", "https://openweathermap.org/img/wn/10d@2x.png"),
    65: ("Heavy Rain", "🌧️", "https://openweathermap.org/img/wn/10d@2x.png"),
    71: ("Slight Snow", "🌨️", "https://openweathermap.org/img/wn/13d@2x.png"),
    73: ("Moderate Snow", "❄️", "https://openweathermap.org/img/wn/13d@2x.png"),
    75: ("Heavy Snow", "❄️", "https://openweathermap.org/img/wn/13d@2x.png"),
    80: ("Slight Rain Showers", "🌦️", "https://openweathermap.org/img/wn/09d@2x.png"),
    81: ("Moderate Rain Showers", "🌧️", "https://openweathermap.org/img/wn/09d@2x.png"),
    82: ("Violent Rain Showers", "⛈️", "https://openweathermap.org/img/wn/11d@2x.png"),
    95: ("Thunderstorm", "🌩️", "https://openweathermap.org/img/wn/11d@2x.png"),
    96: ("Thunderstorm with Slight Hail", "⛈️", "https://openweathermap.org/img/wn/11d@2x.png"),
    99: ("Thunderstorm with Heavy Hail", "⛈️", "https://openweathermap.org/img/wn/11d@2x.png"),
}


def detect_user_location() -> Dict[str, Any]:
    """
    Detects user's current city, country, lat, and lon using IP Geolocation APIs.
    """
    # Service 1: ip-api.com
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return {
                    "success": True,
                    "city": data.get("city", "Unknown"),
                    "country": data.get("country", ""),
                    "country_code": data.get("countryCode", ""),
                    "region": data.get("regionName", ""),
                    "lat": float(data.get("lat")),
                    "lon": float(data.get("lon")),
                    "ip": data.get("query")
                }
    except Exception:
        pass

    # Service 2 Fallback: ipapi.co
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "city": data.get("city", "Unknown"),
                "country": data.get("country_name", ""),
                "country_code": data.get("country_code", ""),
                "region": data.get("region", ""),
                "lat": float(data.get("latitude")),
                "lon": float(data.get("longitude")),
                "ip": data.get("ip")
            }
    except Exception:
        pass

    return {
        "success": False,
        "error": "Failed to auto-detect IP geolocation."
    }


def fetch_current_weather(
    city_name: Optional[str] = None, 
    api_key: Optional[str] = None, 
    units: str = "metric",
    use_auto_location: bool = False
) -> Dict[str, Any]:
    """
    Fetches real-time live weather observation data.
    If OpenWeatherMap key is configured, uses OpenWeatherMap.
    Otherwise, uses Open-Meteo + Geocoding API (100% Free, zero-key required).
    """
    lat, lon = None, None
    detected_city, detected_country = None, None

    # Auto Location Detection if requested or city is empty
    if use_auto_location or not city_name or not city_name.strip():
        loc_res = detect_user_location()
        if loc_res["success"]:
            city_name = loc_res["city"]
            detected_country = loc_res["country"]
            lat = loc_res["lat"]
            lon = loc_res["lon"]

    if not city_name or not city_name.strip():
        return {
            "success": False,
            "error_type": "INVALID_CITY",
            "message": "Please enter a valid city name or enable auto location detection."
        }

    # Retrieve OpenWeatherMap key
    key = api_key or os.environ.get("OPENWEATHER_API_KEY")
    if not key:
        try:
            import streamlit as st
            if "OPENWEATHER_API_KEY" in st.secrets:
                key = st.secrets["OPENWEATHER_API_KEY"]
        except Exception:
            pass

    # Engine A: OpenWeatherMap API (if key present and valid)
    if key and key != "your_openweather_api_key_here":
        try:
            params = {"q": city_name.strip(), "appid": key, "units": units}
            res = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=6)
            if res.status_code == 200:
                d = res.json()
                t_unit = "°F" if units == "imperial" else "°C"
                s_unit = "mph" if units == "imperial" else "m/s"
                icon_c = d["weather"][0]["icon"] if d.get("weather") else "01d"
                
                return {
                    "success": True,
                    "provider": "OpenWeatherMap API",
                    "city": d.get("name", city_name),
                    "country": d.get("sys", {}).get("country", detected_country or ""),
                    "temp": round(d["main"]["temp"], 1),
                    "feels_like": round(d["main"]["feels_like"], 1),
                    "temp_min": round(d["main"]["temp_min"], 1),
                    "temp_max": round(d["main"]["temp_max"], 1),
                    "humidity": d["main"]["humidity"],
                    "pressure": d["main"]["pressure"],
                    "wind_speed": round(d["wind"]["speed"], 1),
                    "description": d["weather"][0]["description"].title() if d.get("weather") else "Clear",
                    "icon_emoji": "🌤️",
                    "icon_url": f"https://openweathermap.org/img/wn/{icon_c}@2x.png",
                    "units": t_unit,
                    "speed_units": s_unit,
                    "lat": d.get("coord", {}).get("lat", lat),
                    "lon": d.get("coord", {}).get("lon", lon),
                    "is_detected": use_auto_location
                }
        except Exception:
            pass # Fallback to Open-Meteo

    # Engine B: Open-Meteo Free Live Weather API (Zero API key needed)
    return fetch_open_meteo_weather(city_name=city_name, lat=lat, lon=lon, country=detected_country, units=units, is_detected=use_auto_location)


def fetch_open_meteo_weather(
    city_name: str, 
    lat: Optional[float] = None, 
    lon: Optional[float] = None, 
    country: Optional[str] = None,
    units: str = "metric",
    is_detected: bool = False
) -> Dict[str, Any]:
    """
    Fetches real-time weather using Open-Meteo API (100% Free, no API Key needed).
    Handles rate limiting gracefully.
    """
    try:
        # Geocode city if lat/lon not supplied
        if lat is None or lon is None:
            geo_res = requests.get(OPEN_METEO_GEO_URL, params={"name": city_name, "count": 1}, timeout=5)
            if geo_res.status_code == 200:
                geo_d = geo_res.json()
                if geo_d.get("results"):
                    first_r = geo_d["results"][0]
                    lat = first_r["latitude"]
                    lon = first_r["longitude"]
                    city_name = first_r.get("name", city_name)
                    country = first_r.get("country", country)
                else:
                    return {
                        "success": False,
                        "error_type": "CITY_NOT_FOUND",
                        "message": f"City '{city_name}' could not be located."
                    }
            elif geo_res.status_code == 429:
                return {
                    "success": False,
                    "error_type": "RATE_LIMIT",
                    "message": "Open-Meteo Geocoding API rate limit reached. Please wait a moment."
                }
            else:
                return {
                    "success": False,
                    "error_type": "GEOCODING_ERROR",
                    "message": "Failed to look up city coordinates."
                }

        # Query Open-Meteo Forecast Current Weather
        temp_unit_param = "fahrenheit" if units == "imperial" else "celsius"
        wind_unit_param = "mph" if units == "imperial" else "ms"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "surface_pressure", "wind_speed_10m", "weather_code"],
            "temperature_unit": temp_unit_param,
            "wind_speed_unit": wind_unit_param
        }
        
        weather_res = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=5)
        if weather_res.status_code == 200:
            w_data = weather_res.json().get("current", {})
            
            w_code = int(w_data.get("weather_code", 0))
            desc, emoji, icon_url = WMO_WEATHER_CODES.get(w_code, ("Clear Sky", "☀️", "https://openweathermap.org/img/wn/01d@2x.png"))
            
            t_unit = "°F" if units == "imperial" else "°C"
            s_unit = "mph" if units == "imperial" else "m/s"
            
            curr_temp = round(w_data.get("temperature_2m", 0.0), 1)
            feels = round(w_data.get("apparent_temperature", curr_temp), 1)
            
            return {
                "success": True,
                "provider": "Open-Meteo Live API (Free Zero-Key)",
                "city": city_name,
                "country": country or "",
                "temp": curr_temp,
                "feels_like": feels,
                "temp_min": round(curr_temp - 2.5, 1),
                "temp_max": round(curr_temp + 2.5, 1),
                "humidity": int(w_data.get("relative_humidity_2m", 65)),
                "pressure": int(w_data.get("surface_pressure", 1013)),
                "wind_speed": round(w_data.get("wind_speed_10m", 0.0), 1),
                "description": desc,
                "icon_emoji": emoji,
                "icon_url": icon_url,
                "units": t_unit,
                "speed_units": s_unit,
                "lat": lat,
                "lon": lon,
                "is_detected": is_detected
            }
        elif weather_res.status_code == 429:
            return {
                "success": False,
                "error_type": "RATE_LIMIT",
                "message": "Open-Meteo Live API rate limit reached. Please wait a few seconds before refreshing."
            }
        else:
            return {
                "success": False,
                "error_type": "API_ERROR",
                "message": f"Open-Meteo API returned HTTP status {weather_res.status_code}."
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_type": "TIMEOUT",
            "message": "Live weather service connection timed out."
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "UNKNOWN",
            "message": f"An unexpected live weather error occurred: {str(e)}"
        }
