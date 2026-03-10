from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

data = joblib.load(MODEL_PATH)

model = data["model"]
scaler = data["scaler"]
feature_order = data["feature_order"]
target_encoder = data["target_encoder"]
categorical_encoders = data["categorical_encoders"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_dict = {}

        for feature in feature_order:

            if feature not in request.form:
                raise Exception(f"Missing form field: {feature}")

            value = request.form[feature]

            if feature in categorical_encoders:
                value = categorical_encoders[feature].transform([value])[0]
            else:
                value = float(value)

            input_dict[feature] = value

        input_df = pd.DataFrame([input_dict])

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        predicted_label = target_encoder.inverse_transform([prediction])[0]

        if predicted_label.lower() == "low":
            color = "green"
        elif predicted_label.lower() == "medium":
            color = "orange"
        else:
            color = "red"

        return render_template(
            "index.html",
            prediction_text=f"Predicted Cardiovascular Risk: {predicted_label}",
            risk_color=color,
            form=request.form
        )

    except Exception as e:
        print("REAL ERROR:", e)
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            form=request.form
        )

if __name__ == "__main__":
    app.run(debug=True)