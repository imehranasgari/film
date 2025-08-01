
# 🧠 Mini Project 3 - Titanic Survival Prediction

## 📌 Problem Statement and Goal

The goal of this mini project is to develop a machine learning model that predicts the survival of Titanic passengers based on available features in the dataset. This is a classic binary classification problem drawn from the famous Kaggle competition.

## 🧪 Solution Approach

This project walks through the full machine learning pipeline:

* Exploratory Data Analysis (EDA)
* Outlier handling using IQR and Z-Score methods
* Data preprocessing and feature engineering
* Training and comparing multiple classification models:

  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
  * XGBoost
* Evaluation using accuracy, classification report, confusion matrix
* Hyperparameter tuning using GridSearchCV
* Cross-validation using K-Fold

## 🧰 Technologies & Libraries

The following libraries and frameworks were used:

* Python
* pandas, numpy
* matplotlib, seaborn (for visualization)
* scikit-learn
* xgboost
* graphviz, pydotplus (for decision tree visualization)

## ⚙️ Installation & Execution Guide

> ⚠️ Not provided in the notebook. Please ensure the required Python packages are installed via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost graphviz pydotplus
```

To run the notebook:

```bash
jupyter notebook "Mini Project 3.ipynb"
```

## 📊 Key Results / Performance

> Detailed numeric performance metrics (accuracy, F1-score, etc.) are evaluated but not summarized in one location. However, evaluation methods such as classification report and confusion matrix are included for each model.

## 🖼️ Screenshots / Sample Outputs

The notebook includes visualizations such as:

* Correlation heatmaps
* Pair plots and distribution plots
* Decision tree diagrams
* Confusion matrix plots

## 💡 Additional Observations

* Multiple outlier detection strategies (IQR and Z-Score) were implemented, showcasing awareness of data cleaning practices.
* Visual storytelling was effectively used to support model explanation.
* While the notebook lacks in-line comments, the structure and modularity of the pipeline are clearly visible.
* Code is organized in logical blocks despite time constraints mentioned by the author.
* The use of cross-validation and hyperparameter tuning adds robustness to the modeling process.
