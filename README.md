# 🎾 AI ATP Match Prediction Model

[![Status](https://img.shields.io/badge/status-in--progress-orange)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning pipeline designed to predict the outcomes of professional ATP tennis matches. This project implements a custom **Elo Rating system** and compares **Gradient Boosting (XGBoost)** with **Deep Learning (LSTM)** architectures.

---

## 📌 Project Overview
Tennis prediction is uniquely challenging due to surface-specific performance and player momentum. This project addresses these variables through:
* **Surface-Aware Elo:** Ratings calculated independently for Clay, Grass, and Hard courts.
* **Leakage Prevention:** A robust data ingestion process that randomizes player ordering to ensure the model learns performance metrics rather than labels.
* **Dual-Model Approach:** Comparing the interpretability of XGBoost with the sequential memory of LSTM networks.

---

## 🛠️ Technical Pipeline

### 1. Data Ingestion & Randomization
* **Timeframe:** Historical match data from 2015–2024.
* **Symmetry:** Uses a random mask to swap "Winner/Loser" into "Player 1/Player 2" columns, creating a balanced target variable (50% wins for P1, 50% for P2) to prevent model bias.

### 2. Feature Engineering
The core of the model's predictive power comes from engineered features:
* **General Elo:** Reflects overall career standing.
* **Surface Elo:** Captures "Surface Specialists" (e.g., higher ratings on Clay vs. Hard).
* **Momentum:** Rolling win rates over a 5-match window to identify players currently in peak form.
* **Ranking:** Incorporation of official ATP rankings with a fallback for unranked players.



### 3. Machine Learning Models
* **XGBoost:** Utilizes `TimeSeriesSplit` to respect the chronological nature of sports data.
* **LSTM:** A baseline Recurrent Neural Network (RNN) that treats matches as sequences, reshaped into `(samples, time_steps, features)`.

---

## 📂 Repository Structure
```text
├── data/
│   ├── raw/            # Annual ATP .csv files (2015-2024)
│   └── processed/      # Cleaned data with engineered Elo features
├── src/
│   ├── process.py      # Merges raw data & randomizes player order
│   ├── features.py     # Elo and Momentum calculation logic
│   ├── train_xgb.py    # XGBoost training & importance plotting
│   └── train_lstm.py   # TensorFlow/Keras LSTM implementation
├── models/             # Saved .keras and .json model weights
└── README.md
