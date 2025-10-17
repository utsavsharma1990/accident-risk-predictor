🚗 Accident Risk Prediction App
A machine learning-powered web application that predicts accident risk based on road characteristics and environmental conditions using LightGBM.

Show Image
Show Image
Show Image
Show Image

🌟 Live Demo
Try the App Here
https://accident-risk-predictor-utsav.streamlit.app/

📋 Table of Contents
Overview
Features
Installation
Usage
Model Details
Project Structure
Dataset
Technologies Used
Contributing
License
Contact
🎯 Overview
This application uses machine learning to predict accident risk on roads based on various factors including:

Road characteristics (type, lanes, curvature)
Environmental conditions (weather, lighting)
Traffic conditions (speed limits, road signs)
Historical accident data
The model achieves an RMSE of ~0.056 on validation data, significantly outperforming baseline predictions.

✨ Features
Interactive UI: Easy-to-use interface with sliders and dropdowns
Real-time Predictions: Instant accident risk assessment
Risk Analysis: Detailed breakdown of contributing risk factors
Visual Indicators: Color-coded risk levels (Low/Medium/High)
Model Caching: Fast predictions after initial model training
Responsive Design: Works on desktop and mobile devices
🚀 Installation
Prerequisites
Python 3.8 or higher
pip package manager
Local Setup
Clone the repository
bash
git clone https://github.com/utsavsharma1990/accident-risk-predictor.git
cd accident-risk-predictor
Create a virtual environment (optional but recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install required packages
bash
pip install -r requirements.txt
Run the application
bash
streamlit run app.py
The app will open in your default browser at http://localhost:8501

📖 Usage
Running Locally
Ensure you have the train.csv file in the project directory
Run the app using streamlit run app.py
The model will automatically train on first run (takes ~20 seconds)
Subsequent runs will load the cached model instantly
Making Predictions
Road Characteristics:
Select road type (urban/rural/highway)
Adjust number of lanes (1-4)
Set road curvature (0-1 scale)
Choose speed limit (25-70 mph)
Environmental Conditions:
Select lighting conditions (daylight/dim/night)
Choose weather (clear/rainy/foggy)
Pick time of day (morning/afternoon/evening)
Additional Factors:
Toggle road signs presence
Indicate if it's a public road
Mark holiday status
Set school season
Enter number of reported accidents
Click "Predict Accident Risk" to get results
Interpreting Results
Risk Score: 0.0 (lowest risk) to 1.0 (highest risk)
Risk Levels:
🟢 Low Risk: 0.0 - 0.25
🟠 Medium Risk: 0.25 - 0.50
🔴 High Risk: 0.50 - 1.0
Risk Factors: Identified contributors to accident risk
🤖 Model Details
Algorithm
Model: LightGBM (Light Gradient Boosting Machine) Regressor
Type: Gradient Boosting Decision Tree
Hyperparameters
python
LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
Performance Metrics
RMSE: ~0.056 on validation set
Baseline RMSE: 0.166 (mean prediction)
Improvement: ~66% reduction in error
Feature Importance
Top contributing features (in order):

Weather conditions
Lighting conditions
Number of reported accidents
Road curvature
Speed limit
📁 Project Structure
accident-risk-predictor/
│
├── app.py                          # Main Streamlit application
├── predicting-road-accident-risk.ipynb  # Model training notebook
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── train.csv                       # Training dataset (add locally)
├── test.csv                        # Test dataset (add locally)
├── accident_risk_model.pkl         # Trained model (generated)
└── submission.csv                  # Predictions (generated)
📊 Dataset
Data Source
Kaggle Playground Series - Season 5, Episode 10

Features (12)
Categorical: road_type, lighting, weather, time_of_day, road_signs_present, public_road, holiday, school_season
Numerical: num_lanes, curvature, speed_limit, num_reported_accidents
Target Variable
accident_risk: Continuous value between 0 and 1
Dataset Statistics
Training samples: 517,754
Test samples: 172,585
No missing values
🛠️ Technologies Used
Frontend: Streamlit
ML Framework: LightGBM
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Model Persistence: Pickle
📦 Requirements
txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Contact
Utsav Sharma - utsav2177@gmail.com

Project Link: https://github.com/utsavsharma1990/accident-risk-predictor

🙏 Acknowledgments
Kaggle for providing the dataset
Streamlit for the amazing web framework
LightGBM team for the efficient ML library
📈 Future Improvements
 Add model explainability (SHAP values)
 Include time-series analysis for accident trends
 Add geographic mapping capabilities
 Implement model retraining pipeline
 Add A/B testing for model versions
 Create API endpoint for predictions
⭐ If you find this project useful, please consider giving it a star!

