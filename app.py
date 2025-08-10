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

# Streamlit UI setup
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🩺 Breast Cancer Detection")
st.write("Adjust the sliders for tumor features to predict if it’s *Benign* or *Malignant*.")

# Sliders for features in 3 columns
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

# Prediction button
if st.button("🔍 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    pred_label = np.argmax(prediction)
    confidence = prediction[0][pred_label] * 100

    # Display result with colored card
    if pred_label == 1:
        st.markdown(
            f"<div style='padding:20px; border-radius:10px; background-color:#d4edda; color:#155724; font-weight:bold; font-size:20px;'>✅ Benign Tumor<br>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='padding:20px; border-radius:10px; background-color:#f8d7da; color:#721c24; font-weight:bold; font-size:20px;'>⚠ Malignant Tumor<br>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )
