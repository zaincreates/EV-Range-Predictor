# ⚡ EV Range Predictor

A machine learning web app that predicts the driving range of an Electric Vehicle based on real-time inputs like battery level, speed, temperature, and number of passengers.

**Built by Zain Abbas** — Mechanical Engineering Student | AI Enthusiast  
Rajalakshmi Engineering College, Tamil Nadu

---

## Live Demo

> Deploy on [Streamlit Cloud](https://share.streamlit.io) and paste your link here

---

## What it does

- Takes 4 inputs: battery %, average speed, outside temperature, passengers
- Predicts estimated driving range in km using a Gradient Boosting ML model
- Shows a live chart of range vs battery drain
- Color-coded alerts (green / yellow / red) based on predicted range

---

## Tech stack

| Layer | Tool |
|-------|------|
| Language | Python 3 |
| ML Model | scikit-learn — Gradient Boosting Regressor |
| Web App | Streamlit |
| Data | Synthetic dataset (physics-based simulation) |
| Deployment | Streamlit Community Cloud (free) |

---

## Project structure

```
ev-range-predictor/
├── generate_data.py     # generates the training dataset
├── train_model.py       # trains and saves the ML model
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md
```

---

## How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/ev-range-predictor.git
cd ev-range-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate the dataset**
```bash
python generate_data.py
```

**4. Train the model**
```bash
python train_model.py
```

**5. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Model performance

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor |
| R² Score | ~0.98 |
| Mean Absolute Error | ~8.4 km |
| Training samples | 1600 |
| Test samples | 400 |

---

## How to deploy free on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select this repo → set main file as `app.py`
5. Click **Deploy** — you get a free public URL in ~2 minutes

---

## Why this project

This project combines my Mechanical Engineering background (specifically from my HEV systems internship) with applied machine learning. Electric vehicle range prediction is a real engineering challenge — battery degradation, aerodynamic drag, thermal management, and payload all affect range. This app simulates those relationships using a data-driven model.

---

## Connect

- LinkedIn: [linkedin.com/in/zain-abbas-baa73b300](https://linkedin.com/in/zain-abbas-baa73b300)
- Email: thezainabbas2007@gmail.com
