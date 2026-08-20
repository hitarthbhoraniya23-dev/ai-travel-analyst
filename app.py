import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Page Config ---
st.set_page_config(page_title="AI Travel Analyst | Executive Suite", page_icon="✈️", layout="wide")

# --- 2. Custom CSS ---
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #1F77B4; 
        font-family: 'Helvetica Neue', sans-serif;
    }
    .profile-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .profile-card img {
        border-radius: 50%;
        width: 75px;
        height: 75px;
        object-fit: cover;
        margin-bottom: 10px;
        border: 3px solid #ffffff;
    }
    div.stButton > button:first-child {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Robust Cached Data Loading (Strictly CSV & Fallback) ---
@st.cache_data
def load_data():
    try:
        # GUARANTEED SAFE: Using read_csv instead of read_excel
        df = pd.read_csv("small_flight_data.csv")
        df.columns = df.columns.str.strip()
        loaded = True
    except Exception:
        # Safe fallback synthetic data if CSV is missing
        np.random.seed(42)
        n = 400
        airlines = np.random.choice(['IndiGo', 'Air India', 'Vistara', 'SpiceJet'], n)
        sources = np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Kolkata'], n)
        stops = np.random.choice([0, 1, 2], n)
        duration = np.random.uniform(2, 15, n).round(1)
        price = np.random.randint(3000, 15000, n) + (stops * 1500) + (duration * 200)
        
        refundable_status = np.random.choice(['Non-Refundable', 'Partially Refundable', 'Fully Refundable'], n, p=[0.5, 0.3, 0.2])
        cancellation_fee = np.where(refundable_status == 'Non-Refundable', np.random.choice([3000, 5000], n),
                                    np.where(refundable_status == 'Partially Refundable', 1500, 0))
        
        df = pd.DataFrame({
            'Airline': airlines,
            'Source': sources,
            'Total_Stops': stops,
            'Duration_Hours': duration,
            'Refund_Policy':refundable_status,
            'Cancellation_Fee': cancellation_fee,
            'Price': price.astype(int)
        })
        loaded = False

    # Clean and convert any price column safely
    for col in df.columns:
        if 'price' in col.lower() or 'cost' in col.lower():
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Refund_Policy' not in df.columns:
        df['Refund_Policy'] = np.random.choice(['Non-Refundable', 'Partially Refundable', 'Fully Refundable'], len(df))
    if 'Cancellation_Fee' not in df.columns:
        df['Cancellation_Fee'] = np.random.choice([0, 1500, 3000, 4500], len(df))
        
    return df, loaded

df, loaded_successfully = load_data()
target_col = 'Price' if 'Price' in df.columns else df.select_dtypes(include=[np.number]).columns[-1]

# --- 4. Interactive Sidebar with Profile & Global Filters ---
with st.sidebar:
    st.markdown("""
        <div class="profile-card">
            <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Hitarth" alt="Avatar">
            <h3 style="margin: 5px 0 0 0; color: white;">Hitarth</h3>
            <p style="font-size: 13px; color: #e0e0e0; margin: 0; font-weight: 600;">Lead AI/ML Analyst Candidate</p>
            <p style="font-size: 11px; color: #b0bec5; margin-top: 4px;">Club Recruitment Portfolio</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("🎛️ Live Data Controls")
    
    airlines_list = df['Airline'].unique().tolist() if 'Airline' in df.columns else []
    selected_airlines = st.multiselect("Filter Airlines", airlines_list, default=airlines_list)
    
    max_price_val = int(df[target_col].max()) if target_col in df.columns else 20000
    price_filter = st.slider("Max Price Threshold (₹)", 3000, max_price_val, max_price_val)
    
    st.divider()
    st.success(f"Active Dataset Rows: {df.shape[0]}")

# Apply filters dynamically
filtered_df = df.copy()
if selected_airlines and 'Airline' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Airline'].isin(selected_airlines)]
if target_col in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[target_col] <= price_filter]

# --- 5. Main Header ---
st.title("✈️ AI Travel Analyst & Predictive Suite")
st.markdown("An enterprise-grade analytics dashboard featuring interactive EDA filtering, cancellation metrics, and live machine learning inference.")
st.divider()

# --- 6. Tabs ---
tab1, tab2 = st.tabs(["📊 Part 1: Interactive Exploration", "🤖 Part 2: ML Predictor & Simulator"])

# ==========================================
# TAB 1: INTERACTIVE EXPLORATION
# ==========================================
with tab1:
    st.subheader("📊 Dynamic Market Intelligence & Policy Breakdown")
    st.markdown("All visualizations below update in real-time based on your sidebar filter selections.")
    
    with st.expander("🔍 Click to Inspect Filtered Dataset View"):
        st.dataframe(filtered_df.head(10), use_container_width=True)
    
    sns.set_theme(style="white")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 1. {target_col} Distribution Density")
        fig1, ax1 = plt.subplots(figsize=(7, 4.2))
        sns.histplot(filtered_df[target_col].dropna(), bins=25, kde=True, ax=ax1, color='#ff6b6b', alpha=0.6)
        ax1.set_xlabel(f"{target_col} (₹)", fontsize=10, fontweight='bold')
        ax1.set_ylabel("Flight Count", fontsize=10, fontweight='bold')
        sns.despine(top=True, right=True)
        st.pyplot(fig1, clear_figure=True)
        
        policy_col = 'Refund_Policy' if 'Refund_Policy' in filtered_df.columns else None
        if policy_col:
            st.markdown(f"#### 3. Refund Policy vs. {target_col}")
            fig3, ax3 = plt.subplots(figsize=(7, 4.2))
            sns.boxplot(x=policy_col, y=target_col, data=filtered_df, ax=ax3, palette='Set2')
            ax3.set_xlabel("Policy Tier", fontsize=10, fontweight='bold')
            ax3.set_ylabel(f"{target_col} (₹)", fontsize=10, fontweight='bold')
            plt.xticks(rotation=10)
            sns.despine(top=True, right=True)
            st.pyplot(fig3, clear_figure=True)

    with col2:
        cat_col = 'Airline' if 'Airline' in filtered_df.columns else None
        if cat_col:
            st.markdown(f"#### 2. Average Fare by {cat_col}")
            fig2, ax2 = plt.subplots(figsize=(7, 4.2))
            avg_grouped = filtered_df.groupby(cat_col)[target_col].mean().sort_values()
            avg_grouped.plot(kind='barh', ax=ax2, color=sns.color_palette("muted", len(avg_grouped) if len(avg_grouped)>0 else 1), edgecolor='none')
            ax2.set_xlabel(f"Average {target_col} (₹)", fontsize=10, fontweight='bold')
            ax2.set_ylabel(cat_col, fontsize=10, fontweight='bold')
            sns.despine(top=True, right=True)
            st.pyplot(fig2, clear_figure=True)
        
        if 'Duration_Hours' in filtered_df.columns:
            st.markdown(f"#### 4. Journey Duration vs. {target_col}")
            fig4, ax4 = plt.subplots(figsize=(7, 4.2))
            sns.scatterplot(x='Duration_Hours', y=target_col, data=filtered_df, alpha=0.7, ax=ax4, color='#1dd1a1', s=60, edgecolor='white')
            ax4.set_xlabel("Duration (Hours)", fontsize=10, fontweight='bold')
            ax4.set_ylabel(f"{target_col} (₹)", fontsize=10, fontweight='bold')
            sns.despine(top=True, right=True)
            st.pyplot(fig4, clear_figure=True)

    st.divider()
    st.markdown("### 📋 Active Portfolio Policy Summary")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        avg_fee = filtered_df['Cancellation_Fee'].mean() if 'Cancellation_Fee' in filtered_df.columns else 0
        st.metric(label="Average Cancellation Fee", value=f"₹{avg_fee:,.2f}")
    with col_r2:
        ref_share = (filtered_df['Refund_Policy'] == 'Fully Refundable').mean() * 100 if 'Refund_Policy' in filtered_df.columns else 0
        st.metric(label="Fully Refundable Share", value=f"{ref_share:.1f}%")
    with col_r3:
        avg_ticket = filtered_df[target_col].mean() if target_col in filtered_df.columns else 0
        st.metric(label="Filtered Average Fare", value=f"₹{avg_ticket:,.2f}")

# ==========================================
# TAB 2: ML PREDICTOR & LIVE SIMULATOR
# ==========================================
with tab2:
    st.subheader("🤖 Machine Learning Engine & Live Fare Predictor")
    st.markdown("Trained using a Random Forest Regressor. Use the interactive simulator below to test custom live predictions!")
    
    @st.cache_resource
    def train_model(data, target):
        df_m = data.dropna(subset=[target]).copy()
        for col in df_m.select_dtypes(include=[np.number]).columns:
            df_m[col] = df_m[col].fillna(df_m[col].median())
        
        encoders = {}
        for col in df_m.select_dtypes(include=['object', 'category']).columns:
            df_m[col] = df_m[col].fillna("Unknown")
            le = LabelEncoder()
            df_m[col] = le.fit_transform(df_m[col].astype(str))
            encoders[col] = le
            
        X = df_m.drop(columns=[target])
        for col in X.columns:
            if 'id' in col.lower() or 'name' in col.lower():
                X = X.drop(columns=[col])
        y = df_m[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return model, X.columns, encoders, r2_score(y_test, preds), mean_absolute_error(y_test, preds)

    try:
        model, feature_cols, encoders, r2, mae = train_model(df, target_col)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Model R² Accuracy Score", value=f"{r2*100:.2f}%")
        with col_m2:
            st.metric(label="Mean Absolute Error (MAE)", value=f"₹{mae:.2f}")
        
        st.divider()
        
        st.markdown("### 🔮 Live Fare Prediction Simulator")
        st.markdown("Test the model instantly by configuring custom flight parameters below:")
        
        sim_col1, sim_col2 = st.columns(2)
        sim_inputs = {}
        
        with sim_col1:
            if 'Airline' in df.columns:
                sim_inputs['Airline'] = st.selectbox("Select Airline", df['Airline'].unique())
            if 'Source' in df.columns:
                sim_inputs['Source'] = st.selectbox("Select Source City", df['Source'].unique())
            if 'Total_Stops' in df.columns:
                sim_inputs['Total_Stops'] = st.selectbox("Total Stops", [0, 1, 2])
                
        with sim_col2:
            if 'Duration_Hours' in df.columns:
                sim_inputs['Duration_Hours'] = st.slider("Duration (Hours)", 1.0, 24.0, 5.0)
            if 'Refund_Policy' in df.columns:
                sim_inputs['Refund_Policy'] = st.selectbox("Refund Policy", df['Refund_Policy'].unique())
            if 'Cancellation_Fee' in df.columns:
                sim_inputs['Cancellation_Fee'] = st.selectbox("Cancellation Fee (₹)", [0, 1500, 3000, 4500])

        if st.button("Calculate Predicted Fare", type="primary"):
            input_df = pd.DataFrame([sim_inputs])
            
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[feature_cols]
            
            for col in input_df.select_dtypes(include=['object', 'category']).columns:
                if col in encoders:
                    le = encoders[col]
                    val = str(input_df[col].iloc[0])
                    if val in le.classes_:
                        input_df[col] = le.transform([val])
                    else:
                        input_df[col] = 0
                        
            predicted_price = model.predict(input_df)[0]
            st.success(f"🎉 **Estimated Flight Ticket Price:** ₹{predicted_price:,.2f}")
            st.balloons()

        st.divider()
        st.markdown("#### 🌟 Key Feature Importance Breakdown")
        
        fig6, ax6 = plt.subplots(figsize=(9, 4.2))
        feat_imps = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imps)))
        
        bars = feat_imps.plot(kind='barh', ax=ax6, color=colors, edgecolor='none')
        for bar in bars.patches:
            width = bar.get_width()
            ax6.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{width:.3f}', 
                     va='center', ha='left', fontweight='bold', fontsize=9, color='#2c3e50')
            
        ax6.set_title("Feature Importance Score Distribution", fontsize=12, fontweight='bold', color='#2c3e50', pad=12)
        ax6.set_xlabel("Relative Importance Weight", fontsize=10, fontweight='bold')
        ax6.set_ylabel("Dataset Attributes", fontsize=10, fontweight='bold')
        ax6.set_xlim(0, max(feat_imps) * 1.20)
        sns.despine(top=True, right=True)
        st.pyplot(fig6, clear_figure=True)
        
    except Exception as e:
        st.error(f"Simulator/Modeling Error: {e}")
