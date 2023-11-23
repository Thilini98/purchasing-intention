from tensorflow import keras
import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pre-trained model
classifier = keras.models.load_model("nn.h5")

def convert(user_input):
    # Split the input string into a list of strings
    numbers_str = user_input.split(',')

    # Convert the list of strings to a NumPy array of floats
    try:
        numbers_array = np.array([float(num) for num in numbers_str])
        return numbers_array.reshape(1, -1)  # Reshape to a 2D array
    except ValueError:
        return None

def predict(X_test):
    y_pred = classifier.predict(X_test)
    y_pred = y_pred > 0.5
    return y_pred.tolist()  # Convert NumPy array to a Python list

@app.route("/", methods=["POST"])
def index():
    try:
        input_data = request.json.get("input")
        if input_data is not None:
            X_data = convert(input_data)
            if X_data is not None:
                output = predict(X_data)
                data = {"prediction": output}
                return jsonify(data)
            else:
                return jsonify({"error": "Invalid input format"})
        else:
            return jsonify({"error": "Missing 'input' key in JSON request"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
