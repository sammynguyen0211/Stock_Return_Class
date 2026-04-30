import os
import sys
import warnings
import tarfile
import tempfile
import posixpath

import boto3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sagemaker
import shap
import streamlit as st
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

warnings.simplefilter("ignore")

# ------------------------------------------------------------
# Page setup


# ------------------------------------------------------------
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

file_path = os.path.join(project_root, "Portfolio", "X_train.csv")

@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    # Drop ALL unnamed/index columns aggressively
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    df = df.loc[:, ~df.columns.str.match(r"^index$", na=False)]
    return df

try:
    dataset = load_dataset(file_path)
except Exception as e:
    st.error(f"Could not load X_train.csv from {file_path}. Error: {e}")
    st.stop()

# ------------------------------------------------------------
# AWS secrets
# ------------------------------------------------------------
try:
    aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
    aws_token = st.secrets["aws_credentials"].get("AWS_SESSION_TOKEN", None)
    aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
    aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
except Exception as e:
    st.error(f"Missing AWS secrets. Check Streamlit secrets. Error: {e}")
    st.stop()

# ------------------------------------------------------------
# AWS session
# ------------------------------------------------------------
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    session_kwargs = {
        "aws_access_key_id": aws_id,
        "aws_secret_access_key": aws_secret,
        "region_name": "us-east-1",
    }
    if aws_token:
        session_kwargs["aws_session_token"] = aws_token
    return boto3.Session(**session_kwargs)

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "shap_explainer.pkl",
    "pipeline": "fraud_model.tar.gz",
    "s3_model_folder": "sklearn-pipeline-deployment",
    "inputs": [
        {"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
        for k in ["C5", "C2", "C1", "C6"]
    ],
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
@st.cache_resource
def load_pipeline(_session, bucket, folder_key):
    s3_client = _session.client("s3")
    local_tar = os.path.join(tempfile.gettempdir(), MODEL_INFO["pipeline"])
    s3_key = posixpath.join(folder_key, os.path.basename(MODEL_INFO["pipeline"]))

    try:
        s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=local_tar)
    except Exception as e:
        st.error(f"Could not download model from s3://{bucket}/{s3_key}. Error: {e}")
        st.stop()

    extract_dir = os.path.join(tempfile.gettempdir(), "streamlit_model_extract")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with tarfile.open(local_tar, "r:gz") as tar:
            names = tar.getnames()
            tar.extractall(path=extract_dir)
    except Exception as e:
        st.error(f"Could not open or extract {MODEL_INFO['pipeline']}. Error: {e}")
        st.stop()

    model_files = [name for name in names if name.endswith((".joblib", ".pkl"))]

    if not model_files:
        st.error(f"No .joblib or .pkl file found inside the tar file. Files found: {names}")
        st.stop()

    model_path = os.path.join(extract_dir, model_files[0])

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Could not load model file {model_files[0]}. Error: {e}")
        st.stop()

@st.cache_resource
def load_shap_explainer(_session, bucket, key):
    s3_client = _session.client("s3")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))

    try:
        if not os.path.exists(local_path):
            s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)
    except Exception as e:
        st.warning(f"Could not download SHAP explainer. SHAP chart will be skipped. Error: {e}")
        return None

    try:
        return joblib.load(local_path)
    except Exception as e:
        st.warning(f"Could not load SHAP explainer. SHAP chart will be skipped. Error: {e}")
        return None


def build_input_dataframe(base_df, user_inputs):
    row = base_df.iloc[[0]].copy()

    for key, value in user_inputs.items():
        if key in row.columns:
            row.loc[row.index[0], key] = value
        else:
            row[key] = value

    return row


def clean_dataframe(df):
    """Remove any unnamed or index columns before sending to endpoint."""
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    df = df.loc[:, ~df.columns.str.match(r"^index$", na=False)]
    return df

def call_model_api(input_df):
    try:
        # Load model locally from S3 (you already built this)
        model = load_pipeline(session, aws_bucket, MODEL_INFO["s3_model_folder"])

        # Clean input
        clean_df = clean_dataframe(input_df)

        # Predict directly
        pred = model.predict(clean_df)
        pred_val = int(pred[0])

        mapping = {0: "Legitimate", 1: "Fraud"}

        return mapping.get(pred_val, str(pred_val)), 200
        
    except Exception as e:
        return f"Prediction error: {e}", 500


def get_preprocessor_and_model(best_pipeline):
    if hasattr(best_pipeline, "named_steps"):
        preprocessor = best_pipeline.named_steps.get("preprocess", None)
        model = best_pipeline.named_steps.get("model", None)
        return preprocessor, model
    return None, best_pipeline


def display_explanation(input_df):
    explainer_key = posixpath.join("explainer", MODEL_INFO["explainer"])
    explainer = load_shap_explainer(session, aws_bucket, explainer_key)

    if explainer is None:
        st.info("Prediction worked, but SHAP explanation was skipped because the explainer could not be loaded.")
        return

    best_pipeline = load_pipeline(session, aws_bucket, MODEL_INFO["s3_model_folder"])
    preprocessor, _ = get_preprocessor_and_model(best_pipeline)

    try:
        if preprocessor is not None:
            transformed = preprocessor.transform(input_df)
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
            explain_df = pd.DataFrame(transformed, columns=feature_names)
        else:
            explain_df = input_df.copy()

        shap_values = explainer(explain_df)
    except Exception as e:
        st.info(f"Prediction worked, but SHAP explanation could not be created. Error: {e}")
        return

    st.subheader("🔍 Decision Transparency (SHAP)")

    try:
        fig = plt.figure(figsize=(10, 4))

        if len(shap_values.shape) == 3:
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            top_values = shap_values[0, :, 1].values
            top_names = shap_values[0, :, 1].feature_names
        else:
            shap.plots.waterfall(shap_values[0], show=False)
            top_values = shap_values[0].values
            top_names = shap_values[0].feature_names

        st.pyplot(fig)

        top_feature = pd.Series(np.abs(top_values), index=top_names).idxmax()
        st.info(f"Business Insight: The most influential factor in this decision was **{top_feature}**.")
    except Exception as e:
        st.info(f"SHAP values were calculated, but the waterfall chart could not be displayed. Error: {e}")

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"],
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    original = build_input_dataframe(dataset, user_inputs)

    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original)
    else:
        st.error(res)
