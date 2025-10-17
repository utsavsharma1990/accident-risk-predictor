import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Fix for pandas compatibility with LightGBM
import pandas.core.indexes.api as indexes_api
if not hasattr(indexes_api, 'Int64Index'):
    indexes_api.Int64Index = indexes_api.Index

from lightgbm import LGBMRegressor

# Page configuration
st.set_page_config(
    page_title="Accident Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Predict Accident Risk")
st.markdown("---")

# Function to train and save model
@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    model_path = 'accident_risk_model.pkl'
    
    # Check if model exists
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    # If model doesn't exist, train it
    try:
        # Load training data
        train = pd.read_csv("train.csv")
        
        # Prepare features
        X = train.iloc[:, 1:-1]
        y = train.iloc[:, -1]
        
        # Define categorical features
        categorical_features = ['road_type', 'lighting', 'weather', 
                              'time_of_day', 'road_signs_present', 'public_road']
        
        # Convert to category dtype
        for col in categorical_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        # Train model
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        model.fit(X, y)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# Load model
model = load_or_train_model()

if model is None:
    st.error("Failed to load or train model. Please ensure train.csv is available.")
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Road Characteristics")
    
    # Road Type
    road_type = st.selectbox(
        "Road Type",
        options=['urban', 'rural', 'highway'],
        help="Select the type of road"
    )
    
    # Number of Lanes
    num_lanes = st.slider(
        "Number of Lanes",
        min_value=1,
        max_value=4,
        value=2,
        help="Select the number of lanes"
    )
    
    # Curvature
    curvature = st.slider(
        "Road Curvature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Road curvature (0 = straight, 1 = highly curved)"
    )
    
    # Speed Limit
    speed_limit = st.slider(
        "Speed Limit (mph)",
        min_value=25,
        max_value=70,
        value=45,
        step=5,
        help="Posted speed limit"
    )
    
    # Road Signs Present
    road_signs_present = st.selectbox(
        "Road Signs Present",
        options=[True, False],
        format_func=lambda x: "Yes" if x else "No",
        help="Are road signs present?"
    )

with col2:
    st.subheader("🌤️ Environmental Conditions")
    
    # Lighting
    lighting = st.selectbox(
        "Lighting Conditions",
        options=['daylight', 'dim', 'night'],
        help="Select lighting conditions"
    )
    
    # Weather
    weather = st.selectbox(
        "Weather Conditions",
        options=['clear', 'rainy', 'foggy'],
        help="Select weather conditions"
    )
    
    # Time of Day
    time_of_day = st.selectbox(
        "Time of Day",
        options=['morning', 'afternoon', 'evening'],
        help="Select time of day"
    )
    
    # Public Road
    public_road = st.selectbox(
        "Public Road",
        options=[True, False],
        format_func=lambda x: "Yes" if x else "No",
        help="Is this a public road?"
    )
    
    # Holiday
    holiday = st.selectbox(
        "Holiday",
        options=[True, False],
        format_func=lambda x: "Yes" if x else "No",
        help="Is it a holiday?"
    )
    
    # School Season
    school_season = st.selectbox(
        "School Season",
        options=[True, False],
        format_func=lambda x: "Yes" if x else "No",
        help="Is it during school season?"
    )
    
    # Number of Reported Accidents
    num_reported_accidents = st.slider(
        "Number of Reported Accidents",
        min_value=0,
        max_value=5,
        value=1,
        help="Number of previously reported accidents at this location"
    )

st.markdown("---")

# Predict button
if st.button("🔍 Predict Accident Risk", type="primary", use_container_width=True):
    # Create input dataframe
    input_data = pd.DataFrame({
        'road_type': [road_type],
        'num_lanes': [num_lanes],
        'curvature': [curvature],
        'speed_limit': [speed_limit],
        'lighting': [lighting],
        'weather': [weather],
        'road_signs_present': [road_signs_present],
        'public_road': [public_road],
        'time_of_day': [time_of_day],
        'holiday': [holiday],
        'school_season': [school_season],
        'num_reported_accidents': [num_reported_accidents]
    })
    
    # Convert categorical features to category dtype
    categorical_features = ['road_type', 'lighting', 'weather', 
                          'time_of_day', 'road_signs_present', 'public_road']
    
    for col in categorical_features:
        input_data[col] = input_data[col].astype('category')
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.markdown("### Prediction Result")
        
        # Create columns for result display
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col2:
            # Determine risk level
            if prediction < 0.25:
                risk_level = "Low Risk"
                color = "green"
                emoji = "✅"
            elif prediction < 0.5:
                risk_level = "Medium Risk"
                color = "orange"
                emoji = "⚠️"
            else:
                risk_level = "High Risk"
                color = "red"
                emoji = "🚨"
            
            # Display prediction
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h1 style='color: {color}; margin: 0;'>{emoji} {risk_level}</h1>
                    <h2 style='margin: 10px 0;'>Accident Risk Score: {prediction:.3f}</h2>
                    <p style='color: #666; margin: 0;'>Risk scale: 0.0 (lowest) to 1.0 (highest)</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(min(prediction, 1.0))
            
            # Additional information
            st.markdown("---")
            st.markdown("#### Risk Factors Analysis")
            
            factors = []
            if lighting == 'night':
                factors.append("🌙 Night driving increases risk")
            if weather in ['rainy', 'foggy']:
                factors.append(f"🌧️ {weather.capitalize()} conditions increase risk")
            if curvature > 0.7:
                factors.append("🛣️ High road curvature increases risk")
            if speed_limit >= 60:
                factors.append("⚡ High speed limit increases risk")
            if not road_signs_present:
                factors.append("🚫 Absence of road signs increases risk")
            if num_reported_accidents > 2:
                factors.append("⚠️ Multiple previous accidents at this location")
            
            if factors:
                for factor in factors:
                    st.markdown(f"- {factor}")
            else:
                st.success("✅ No major risk factors identified")
                
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts accident risk based on various road and environmental factors.
    
    **Model:** LightGBM Regressor
    
    **Features:**
    - Road characteristics
    - Environmental conditions
    - Historical accident data
    
    **Risk Scale:**
    - 0.0 - 0.25: Low Risk
    - 0.25 - 0.5: Medium Risk
    - 0.5 - 1.0: High Risk
    """)
    
    st.markdown("---")
    st.markdown("**Note:** This is a predictive model and should be used for reference only.")