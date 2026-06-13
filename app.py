from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("breast_cancer_model", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "Malignant (Cancer Detected)"
        else:
            result = "Benign (No Cancer)"

        return render_template(
            "index.html",
            prediction_text=f"Prediction Result: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)