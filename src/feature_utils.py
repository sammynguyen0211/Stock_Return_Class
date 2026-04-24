
import json

def convert_input(request_body, request_content_type):
    """
    Convert incoming request into model-ready CSV format.

    Expected JSON input:
    {
        "sentiment_textblob": float,
        "sentiment_lag1": float
    }

    Output:
    CSV string → "value1,value2"
    """

    data = json.loads(request_body)

    sentiment_textblob = data["sentiment_textblob"]
    sentiment_lag1 = data["sentiment_lag1"]

    # Return CSV string (must match training feature order)
    csv_input = f"{sentiment_textblob},{sentiment_lag1}"

    return csv_input


def model_fn(model_dir):
    """
    Load trained model from disk
    """
    import os
    from joblib import load

    model_path = os.path.join(model_dir, "finalized_sentiment_model.joblib")
    model = load(model_path)

    return model


def predict_fn(input_data, model):
    """
    Run prediction using loaded model
    """
    import numpy as np

    # Convert CSV string → array
    values = np.array([float(x) for x in input_data.split(",")]).reshape(1, -1)

    prediction = model.predict(values)

    return prediction


def output_fn(prediction, content_type):
    """
    Format output as JSON
    """
    return json.dumps({"prediction": prediction.tolist()})
