# 🚆 Rainlyzer – Understand the rain forecast for tomorrow in seconds.

## 🔍 Project Overview

This project presents a machine learning-based approach to predict rail fall incidents by analyzing structured infrastructure and environmental data. Multiple classification algorithms were explored, with the **CatBoost classifier** demonstrating the highest performance in terms of accuracy and reliability.

To make the model accessible and interactive, it is integrated into a **Streamlit web application**, allowing users to perform real-time predictions based on input features. The solution showcases how AI can be applied effectively to classify and anticipate potential failures in rail-related systems using historical and sensor-derived data.

---

## 🛠️ Project Workflow

### 1. Data Preprocessing
- Handled missing values and outliers
- Encoded categorical features
- Scaled numerical features
- Balanced the dataset using SMOTE (Synthetic Minority Oversampling Technique)

### 2. Model Building
- Compared multiple machine learning models:
  - CatBoost (Best Performance)
  - Random Forest
  - XGBoost
  - Logistic Regression
  - K-Nearest Neighbors
  - Naive Bayes
  - Support Vector Machine

### 3. Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report

### 4. Deployment
- Saved the best-performing model (`best_catboost_model.pkl`)
- Deployed using Streamlit for user interaction and prediction

---

## 📦 Tech Stack

- **Languages**: Python
- **Libraries**: `scikit-learn`, `catboost`, `xgboost`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `imbalanced-learn`
- **Deployment**: Streamlit

---

## 📁 Project Structure


## ✅ Features

- Robust model performance with SMOTE
- Simple web interface using Streamlit
- Exported model for reuse or integration

---

## 🚀 Future Enhancements

- Real-time prediction using live sensor data
- Integration of LSTM for time-series analysis
- Notification system for high-risk alerts

## 📬 Contact

<p>
  <a href="mailto:aradhyaray99@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" /></a>
  <a href="www.linkedin.com/in/rayaradhya"><img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
  <a href="https://github.com/AradhyaRay05"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
</p>

---

Thanks for visiting ! Feel free to explore my other repositories and connect with me. 🚀 
