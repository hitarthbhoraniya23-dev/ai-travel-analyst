# ✈️ AI Travel Analyst
**MIC AIML Recruitment Challenge — 2nd Year Data Science Track**

## 📌 Project Overview
The AI Travel Analyst is an interactive data dashboard built to explore historical flight data and predict future ticket prices. It features comprehensive Exploratory Data Analysis (EDA) and a machine learning regression model.

## 📊 Features
* **Part 1 (Data Exploration):** Cleans dataset missing values and renders 5 distinct visualizations (Histograms, Bar Charts, Box Plots, Scatter Plots, and a Correlation Heatmap) to uncover pricing trends.
* **Part 2 (Price Prediction):** Utilizes `LabelEncoder` for categorical variables and trains a `RandomForestRegressor` to estimate flight costs.
* **Model Insights:** Displays R² accuracy scores, Mean Absolute Error (MAE), and a Feature Importance chart to explain the model's logic.

## 🚀 Installation & Execution
1. Clone this repository to your local machine.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit Dashboard: `python -m streamlit run app.py`