import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load dataset
cancer_data = datasets.load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
feature_names = cancer_data.feature_names

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build and train model
tf.random.set_seed(42)
model = keras.Sequential([
    keras.layers.InputLayer(shape=(X.shape[1],)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_scaled, Y, epochs=10, verbose=0)

# Streamlit Page Config
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.markdown(
    """
    <style>
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        animation: fadeIn 0.8s ease-in-out;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Breast Cancer Detection")
st.write("Adjust the sliders for tumor features to predict if it’s **Benign** or **Malignant**.")

# Sliders for features
user_input = []
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    col_idx = i % 3
    min_val = float(X[:, i].min())
    max_val = float(X[:, i].max())
    default_val = float(X[:, i].mean())
    with cols[col_idx]:
        val = st.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=(max_val - min_val) / 100
        )
    user_input.append(val)

# Prediction
if st.button("🔍 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    pred_label = np.argmax(prediction)
    confidence = prediction[0][pred_label] * 100

    if pred_label == 1:
        st.markdown(
            f"<div class='result-card benign'>✅ Benign Tumor<br>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card malignant'>⚠️ Malignant Tumor<br>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

# Disclaimer
st.markdown(
    "<br><b>Disclaimer:</b> This tool is for educational purposes only and is not a substitute for professional medical advice.",
    unsafe_allow_html=True
)
