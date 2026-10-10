# 🌤️ Weather Pattern Visualizer

A complete, modern, soft-pastel-themed Weather Data Analytics & Pattern Visualization Web Application built with **Python**, **Pandas**, **Seaborn**, **Plotly**, **Scikit-Learn**, and **Streamlit**.

Designed specifically as an academic BTech Computer Science & Engineering (CSE) minor project demonstration.

---

## 🎨 Soft Pastel Theme & Design System

The application strictly implements a minimalist, modern, soft pastel UI palette:

- **Background:** `#F8F7FC`
- **Card Background:** `#FFFFFF`
- **Borders & Dividers:** `#E7E5EF`
- **Primary Pastel Blue:** `#C9E4F6` (Accent: `#5C90D2`)
- **Pastel Lavender:** `#E5DDF8` (Accent: `#9B82C9`)
- **Pastel Mint Green:** `#D9F2E6` (Accent: `#48B188`)
- **Pastel Peach:** `#FFE3D5` (Accent: `#E88564`)
- **Pastel Yellow:** `#FFF2C6` (Accent: `#E5C158`)
- **Primary Text:** `#343447`
- **Secondary Text:** `#73738A`

---

## ✨ Application Features

1. **📊 Overview Dashboard:** Dynamic summary cards calculating average temperature, maximum/minimum recorded temperatures, total precipitation, average humidity, and total record count.
2. **📂 Historical Analysis:** Smart fuzzy CSV column detection, automatic data hygiene validation, date parsing, missing value handling, and dataset preview.
3. **🌡️ Temperature Trends:** Interactive Plotly daily time-series with 7-day moving averages, monthly temperature bars, min-max temperature range bands, Seaborn statistical KDE distributions, boxplots, and automated trend explanations.
4. **🌧️ Precipitation Analysis:** Daily rainfall charts, monthly totals highlighting wettest/driest months, and rainy days counting.
5. **💨 Humidity & Wind Dynamics:** Subplot line charts, temperature vs. humidity scatter plots with OLS regression trendlines, rainfall vs. humidity intensity charts, and multivariate correlation heatmaps.
6. **🏙️ Weather Comparison:** Side-by-side grouped bar charts and statistical comparison tables across multiple cities.
7. **🔍 Data Explorer:** Filterable dataset table with search query capabilities, column selectors, descriptive statistics (`df.describe()`), and single-click CSV dataset export.
8. **🌤️ Live Current Weather (API):** OpenWeatherMap REST API integration for real-time observation lookups by city name.
9. **📈 Temperature Trend Prediction (AI/ML):** Scikit-Learn Linear & Polynomial Regression models fitting historical dates to estimate future temperature trend directions with evaluation metrics ($R^2$, MAE, RMSE).
10. **ℹ️ About Project:** Full academic presentation documentation, problem statement, technology stack, and college viva Q&A notes.

---

## 📁 Project Folder Structure

```
minor project/
├── .streamlit/
│   ├── config.toml           # Streamlit theme & server configuration
│   └── secrets.toml.example   # Example secrets file for API key setup
├── data/
│   ├── sample_weather.csv     # 1-Year daily weather dataset (Seattle)
│   └── sample_multi_city.csv  # 1-Year multi-city weather dataset (NY, London, Tokyo)
├── utils/
│   ├── __init__.py            # Python package init
│   ├── data_processing.py     # Fuzzy column mapping, cleaning, metrics & insights
│   ├── visualizations.py      # Pastel Plotly & Seaborn visualizers
│   ├── weather_api.py         # OpenWeatherMap API wrapper
│   └── ml_prediction.py       # Scikit-learn trend modeling & forecasting
├── generate_sample_data.py    # Script used to generate sample datasets
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependency requirements
├── render.yaml                # Render Web Service blueprint configuration
├── Procfile                   # Procfile for web server execution
├── .gitignore                 # Files excluded from git
└── README.md                  # Comprehensive project documentation
```

---

## 🚀 Local Installation & Execution

### Prerequisites
- Python 3.10+ installed on your machine.

### Step-by-Step Setup

1. **Clone / Open Project Directory:**
   ```bash
   cd "minor project"
   ```

2. **Create a Virtual Environment:**
   - **Windows:**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

5. **Access in Browser:**
   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 OpenWeatherMap API Setup (Optional)

The application includes sample historical datasets out-of-the-box and does **not** require an API key to run basic historical visual analysis.

To enable real-time weather lookup:
1. Register for a free API key at [openweathermap.org/api](https://openweathermap.org/api).
2. Option A: Enter the API key directly into the sidebar expander in the app interface **"🔑 OpenWeatherMap API Key"**.
3. Option B: Create a `.streamlit/secrets.toml` file with:
   ```toml
   OPENWEATHER_API_KEY = "your_actual_api_key_here"
   ```

---

## ☁️ Deployment Guide (Render Web Service)

This application is ready for zero-configuration deployment on **Render**.

### Step 1: Push Code to GitHub
1. Initialize Git and commit files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Weather Pattern Visualizer"
   ```
2. Create a new repository on GitHub and push code:
   ```bash
   git remote add origin https://github.com/your-username/weather-pattern-visualizer.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Create a Render Web Service
1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Fill in the service configuration:
   - **Name:** `weather-pattern-visualizer`
   - **Environment:** `Python 3`
   - **Region:** Choose nearest region (e.g. Oregon, Frankfurt, Singapore).
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```

### Step 3: Configure Environment Variables (Optional)
If you want to provide a default OpenWeatherMap API key:
- Add Key: `OPENWEATHER_API_KEY`
- Value: `your_openweather_api_key_here`

### Step 4: Deploy & Verify
- Click **Create Web Service**.
- Wait for Render to build dependencies and launch Streamlit.
- Once finished, Render will provide your live application URL (e.g., `https://weather-pattern-visualizer.onrender.com`).

---

## 🎓 Academic Viva Q&A Guide (BTech CSE)

1. **Q: What design pattern is used for separating UI and data processing?**  
   *A: Modular design. `app.py` serves as the Streamlit controller/view layer, while business logic, data cleaning, visualizations, and ML models are decoupled into `utils/data_processing.py`, `utils/visualizations.py`, and `utils/ml_prediction.py`.*

2. **Q: How does fuzzy column mapping work?**  
   *A: The system inspects uploaded headers against dictionary lists of aliases (`date`, `tavg`, `precip`, `humidity`, etc.) using lower-cased normalization and substring matching.*

3. **Q: What regression algorithms are used for trend prediction?**  
   *A: Scikit-learn's `LinearRegression` and degree-2 `PolynomialFeatures` transformation. The model fits relative day indices against observed temperatures and evaluates goodness of fit using $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).*

---

## 📄 License & Attribution
Developed for educational and academic project presentation purposes. Free to modify and adapt.
