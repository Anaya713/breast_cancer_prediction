# breast_cancer_prediction
A Streamlit-based machine learning web app that predicts whether a breast tumor is benign or malignant using a trained neural network model on the Wisconsin Breast Cancer dataset.
Here’s a full README template you can use and customize as needed:

---

# breast\_cancer\_prediction

A **Streamlit-based machine learning web app** that predicts whether a breast tumor is **benign or malignant** using a trained neural network model on the **Wisconsin Breast Cancer dataset**.
This interactive app lets users adjust tumor feature sliders to get instant predictions with confidence scores.

## Features

* User-friendly slider inputs for 30 tumor features
* Real-time prediction of tumor type (benign or malignant)
* Confidence percentage displayed with prediction
* Pretrained TensorFlow Keras model for fast inference
* Clean, responsive UI with 3-column layout

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/breast_cancer_prediction.git
cd breast_cancer_prediction
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and interact with the app.

## Model Training

The neural network model was trained on the Wisconsin Breast Cancer dataset using TensorFlow Keras.
Training code and notebook are available in `breastcancertraining.ipynb`.



If you want me to tailor it further or add screenshots or demo links, just say! 😊
