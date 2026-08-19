import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Basic Page Config (No Custom CSS to prevent conflicts) ---
st.set_page_config(page_title="AI Travel Analyst", page_icon="✈️", layout="wide")

# --- 2. Safety Check Header ---
st.title("✈️ AI Travel Analyst Dashboard")
st.success("✅ SYSTEM ONLINE: If you can see this message, Streamlit is working perfectly!")
st.write("Analyze flight prices, uncover factors driving costs, and predict future fares.")
st.divider()

# --- 3. Bulletproof Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/flight_data.xlsx")
        # Ensure 'Price' column exists before dropping NAs
        if 'Price' in df.columns:
            df = df.dropna(subset=['Price']) 
        return df, True
    except FileNotFoundError:
        # Fallback synthetic data
        np.random.seed(42)
        n = 200
        data = {
            'Airline': np.random.choice(['IndiGo', 'Air India', 'Vistara', 'SpiceJet'], n),
            'Source': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Kolkata'], n),
            'Total_Stops': np.random.choice([0, 1, 2], n),
            'Duration_Hours': np.random.uniform(2, 15, n).round(1),
            'Price': np.random.randint(3000, 15000, n)
        }
        df = pd.DataFrame(data)
        df['Price'] += (df['Total_Stops'] * 1500) + (df['Duration_Hours'] * 200)
        return df, False

# Load data safely
df, loaded_successfully = load_data()

# --- 4. Sidebar ---
with st.sidebar:
    st.header("Admin Panel")
    if loaded_successfully:
        st.success("Dataset Loaded: flight_data.csv")
    else:
        st.warning("Using Synthetic Data (CSV not found)")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

# --- 5. Tabs ---
tab1, tab2 = st.tabs(["📊 Part 1: Exploration", "🤖 Part 2: Prediction Model"])

# ==========================================
# TAB 1: EXPLORATION (5 Visualizations)
# ==========================================
with tab1:
    st.subheader("Data Overview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Use columns to organize charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**1. Price Distribution**")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Price'], bins=20, kde=True, ax=ax1)
        st.pyplot(fig1)
        
        if 'Total_Stops' in df.columns:
            st.write("**3. Stops vs Price**")
            fig3, ax3 = plt.subplots()
            sns.boxplot(x='Total_Stops', y='Price', data=df, ax=ax3)
            st.pyplot(fig3)

    with col2:
        if 'Airline' in df.columns:
            st.write("**2. Average Price by Airline**")
            fig2, ax2 = plt.subplots()
            avg_price = df.groupby('Airline')['Price'].mean().sort_values()
            avg_price.plot(kind='barh', ax=ax2, color='skyblue')
            st.pyplot(fig2)
            
        if 'Duration_Hours' in df.columns:
            st.write("**4. Duration vs Price**")
            fig4, ax4 = plt.subplots()
            sns.scatterplot(x='Duration_Hours', y='Price', data=df, alpha=0.5, ax=ax4)
            st.pyplot(fig4)

    st.write("**5. Correlation Heatmap**")
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    numeric_cols = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='Blues', ax=ax5)
    st.pyplot(fig5)

# ==========================================
# TAB 2: MODELING
# ==========================================
with tab2:
    st.subheader("Random Forest Price Predictor")
    
    try:
        # Simple feature engineering
        df_model = df.copy()
        for col in df_model.select_dtypes(include=['object']).columns:
            df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))
            
        X = df_model.drop('Price', axis=1)
        y = df_model['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        st.write(f"- **R² Score:** {r2_score(y_test, preds)*100:.2f}%")
        st.write(f"- **Mean Absolute Error:** ₹{mean_absolute_error(y_test, preds):.2f}")
        
        st.write("**Feature Importance**")
        fig6, ax6 = plt.subplots()
        feat_imps = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
        feat_imps.plot(kind='barh', ax=ax6, color='coral')
        st.pyplot(fig6)
        
    except Exception as e:
        st.error(f"Modeling Error: {e}")