# Credit-Card-Fraud-Detection
This project aims to build a robust Credit Card Fraud Detection Model using Machine Learning (Random Forest Classifier), starting from Exploratory Data Analysis (EDA) to deployment via Flask. The workflow is based on the famous Kaggle Credit Card Fraud Detection dataset.

## Problem Statement
Credit card fraud is a growing issue for financial institutions.
Goal: Accurately classify transactions as Fraudulent (1) or Legitimate (0).

Dataset contains 30 features:
-Time
-V1 to V28 (anonymized PCA features)
-Amount
-Target: Class (0/1)

## Project Workflow

### 1. Data Understanding & Exploration
- Explored the **Kaggle Credit Card Fraud dataset**.
- Analyzed **class imbalance (0 → Legitimate, 1 → Fraudulent)**.
- Used **histograms**, **boxplots**, and **correlation heatmaps** for feature distribution insights.
- Detected outliers in features like `V3`, `V4`, `V9`, `V10`, `V11`, etc.

### 2. Feature Engineering
- Created new features like:
  - **Transaction Hour** (from `Time`)
  - **Log-transformed Amount** (to reduce skewness)
  - **Outlier Flags** based on boxplot outlier thresholds for key `V` features.
- Dropped raw `Time` and `Amount` columns after transformation.

### 3. Handling Imbalanced Data
- Used two approaches:
  - **Undersampling** (using `RandomUnderSampler` from `imblearn`)
  - **Oversampling** (using `SMOTE`)
- Created separate datasets for both approaches and compared model performance.

### 4. Model Building & Evaluation
- Tried 4 models on Normal, Undersampled, and Oversampled data:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **XGBoost**

 - Evaluation metrics used:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1 Score**
    - **AUC ROC**
    - **Precision-Recall Curve**
    - **Feature Importance (Random Forest & XGBoost)**

**Final Model Chosen**: ✅ **Random Forest on Oversampled Data**

### 5. Model Deployment (Flask + ngrok)
- Created a **Random Forest pipeline** that includes all preprocessing steps:
  - Feature creation
  - Transformation
  - Outlier flags
- Saved using **Joblib** as `rf_credit_fraud_pipeline.pkl`.

- Built a **Flask web app** for user inputs (raw 30 features as in original dataset: Time, V1-V28, Amount).

- Setup **ngrok** for public URL tunneling.

### 6. Demo Screenshots
- **Model Input Page**

![Web UI](./images/web_ui.png)

- **Prediction Output**

![Prediction Result](./images/prediction_output.png)
