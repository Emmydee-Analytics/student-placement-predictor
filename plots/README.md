# Student Placement Predictor: Machine Learning Pipeline

An end-to-end data science portfolio project analyzing student performance factors and predicting job placement status using machine learning.

## 📌 Project Overview
This repository features a predictive classification model that evaluates how a student's academic history (`previous_score`, `exam_score`) and lifestyle traits (`study_hours`, `attendance`, `sleep_hours`, `internet_usage`) impact their final employment placement outcome.

## 📂 Repository Structure
* `main.py` - Core Python script managing preprocessing, modeling, and chart generation.
* `data.csv` - The underlying structured academic performance dataset.
* `plots/` - Exported evaluation charts and feature correlation graphics.

## 📊 How to Run the Project
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Run the automated modeling pipeline:
   ```bash
   python main.py
   ```

## 📉 Machine Learning Results Summary
* **Model Used**: Random Forest Classifier
* **Primary Objective**: Accurately predict whether a student achieves a "Placed" or "Not Placed" status based on behavioral features.
* **Key Insights**: The feature importance plot indicates which factors (such as attendance rates or exam scores) hold the highest statistical weight in determining final employment placement outcomes.

