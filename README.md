# cardio-risk-prediction

Overview

The Cardio Risk Prediction System is a machine learning–based web application that predicts a person's cardiovascular risk level using medical and lifestyle data.
The system uses a trained machine learning model and a Flask web interface to allow users to input health parameters and receive a prediction of their heart disease risk category.

The model classifies risk into three levels:

Low Risk

Medium Risk

High Risk

This project demonstrates the integration of Machine Learning, Data Processing, and Web Deployment using Flask.

Features

Machine learning–based cardiovascular risk prediction

Logistic Regression classification model

Data preprocessing with feature scaling and encoding

Web interface built with Flask

User-friendly input form for health parameters

Risk prediction displayed instantly

Technologies Used

Python

Flask

Scikit-learn

Pandas

NumPy

Joblib

HTML / CSS

Project Structure
cardio-risk-prediction
│
├── app.py
├── trained_model.py
├── model.pkl
├── cardiovascular_risk_dataset.csv
│
├── templates
│   └── index.html
│
└── static
    └── cardiacimg.png
How the System Works
1. Model Training

The model is trained using the dataset:

cardiovascular_risk_dataset.csv

Steps involved:

Data preprocessing

Encoding categorical variables

Feature scaling using StandardScaler

Logistic Regression model training

Saving the trained model using Joblib

The trained model is stored as:

model.pkl
2. Web Application

The Flask application (app.py) loads the trained model and provides a web interface where users can input health-related parameters such as:

Age

Blood pressure

Cholesterol level

Smoking status

Family history of heart disease

Other medical indicators

The model processes the inputs and predicts the cardiovascular risk category.

Installation
1. Clone the Repository
git clone https://github.com/aasim226/cardio-risk-prediction.git
2. Navigate to the Project Folder
cd cardio-risk-prediction
3. Install Required Libraries
pip install -r requirements.txt
Run the Application

Start the Flask server:

python app.py

Then open your browser and go to:

http://127.0.0.1:5000

You can now use the interface to predict cardiovascular risk.

Example Workflow

User enters health information.

Data is processed by the backend.

Machine learning model predicts the risk category.

The predicted risk is displayed on the web page.

Future Improvements

Add deep learning models for improved prediction

Deploy the application online

Integrate real medical datasets

Add visual dashboards for risk analysis

Improve UI/UX design

Author

Aasim Ahmed

Machine Learning and AI Developer
Focused on AI applications in healthcare and intelligent systems.
