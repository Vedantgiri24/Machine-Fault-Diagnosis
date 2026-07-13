# Machine Fault Diagnosis Using Deep Learning

A CNN-based Streamlit web app that classifies machine vibration signal images into one of five conditions: **Bearing Fault**, **Bent Shaft**, **Foundation Looseness**, **Healthy**, or **Misalignment**. Built for predictive maintenance and condition monitoring.

## Features

- Upload a vibration signal graph image (PNG/JPG) from sensor channels CH1, CH2, or CH3
- CNN model (TensorFlow/Keras) predicts the fault class with a confidence score
- Displays fault explanation and recommended maintenance action
- Shows raw softmax probabilities and preprocessed tensor details for technical review

## Project Structure

```
mfd/
└── deploy_package/
    ├── app.py              # Streamlit application
    ├── requirements.txt    # Python dependencies
    ├── runtime.txt         # Pinned Python version (3.11)
    └── best_model.keras    # Trained CNN model
```

## Running Locally

```bash
git clone https://github.com/<your-username>/mfd.git
cd mfd/deploy_package
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub with the structure above.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing to `deploy_package/app.py`.
3. In **Advanced settings**, explicitly select **Python 3.11** before deploying (TensorFlow does not yet support the newer Python versions Streamlit Cloud may default to).

## Model Details

| Property | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Input size | 256 × 128 (W × H), grayscale |
| Classes | 5 (Bearing Fault, Bent Shaft, Foundation Looseness, Healthy, Misalignment) |
| Output | Softmax probabilities |

## Tech Stack

- Streamlit
- TensorFlow (CNN)
- NumPy

## Author

Final Year Project, 2026–27
