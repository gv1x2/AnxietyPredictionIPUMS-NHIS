# AnxietyPredictionIPUMS-NHIS


# 🧠 Anxiety Prediction Using Machine Learning and Neural Networks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Overview
This project applies **machine learning** and **deep learning** techniques to predict **anxiety (ANXIETYEV)** using survey and demographic data from the **IPUMS NHIS** dataset.  
It demonstrates data preprocessing, encoding, model training, and evaluation under class imbalance conditions.

---

## 🎯 Objectives
- Predict anxiety levels based on demographic and health data.  
- Compare classical ML models and neural networks.  
- Manage data imbalance using **SMOTE** and **undersampling**.  
- Interpret results through **feature importance** analysis.

---

## 🧩 Data Preparation
Implemented in **`Database1.py`**

- Recodes variables such as `AGE`, `SEX`, `SEXORIEN`, `MARSTLEG`, `EDUC`, `POVERTY`, `OWNERSHIP`, `HOUYRSLIV`, `HEALTH`, `HEIGHT`, `WEIGHT`.  
- Collapses **LGBTQ+ categories** into one group to increase analytical power.  
- Produces the final dataset `Book27.csv`.

---

## 📊 Exploratory Data Analysis
Implemented in **`module1.py`**

- Correlation and pair plots using **Seaborn**.  
- Visualization examples:
  - Boxplot: Health vs. Anxiety  
  - Violin Plot: Weight vs. Anxiety  
  - Bar Chart: Average Height by Anxiety Group  

---

## 🧠 Modeling Pipeline

| Module | Model | Method Highlights |
|:-------:|:------|:------------------|
| `module2.py` | Logistic Regression | SMOTE balancing and threshold tuning |
| `module3.py` | Bagging (Decision Trees) | Ensemble variance reduction |
| `module4.py` | XGBoost | Gradient boosting with `scale_pos_weight` |
| `module5.py` | Stacking Classifier | Combines Random Forest + Gradient Boosting + Logistic Regression |
| `module6.py` | Neural Network (Keras) | Dense layers with dropout and AUC monitoring |
| `module7.py` | Logistic Regression | Random undersampling for balance |
| `module8.py` | Feature Importance | SHAP, Logistic Regression, and Random Forest insights |

---

## ⚙️ Technologies Used
- **Python Libraries**  
  `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `imblearn`, `matplotlib`, `seaborn`, `shap`

- **Evaluation Metrics**  
  Accuracy, Precision, Recall, F1-score, ROC-AUC, AUC-PR

---

## 🔑 Key Variables
| Variable | Description |
|-----------|--------------|
| **ANXIETYEV** | Target: ever diagnosed with anxiety (0/1) |
| **SEXORIEN** | Collapsed to 0 = Straight, 1 = LGBTQ+ |
| **MARSTLEG** | Legal marital status (federally consistent) |
| **EDUC** | Education grouped into 5 levels |
| **POVERTY** | Income-to-poverty ratio |
| **HEALTH** | Self-rated health |
| **HEIGHT / WEIGHT** | Normalized physical measures |
| **OWNERSHIP / HOUYRSLIV** | Housing context indicators |

---

## 📈 Model Evaluation
Each module prints:
- Confusion Matrix  
- Classification Report  
- ROC-AUC  
- Average Precision (AUC-PR)

Models are compared under both **balanced** (SMOTE/undersampling) and **original** distributions.

---

## 🧾 File Structure
Database1.py # Data preprocessing and encoding
module1.py # Exploratory data analysis
module2.py # Logistic regression with SMOTE
module3.py # Bagging classifier
module4.py # XGBoost model
module5.py # Stacking / AdaBoost models
module6.py # Neural network
module7.py # Undersampling + Logistic regression
module8.py # Feature importance analysis
Book27.csv # Processed dataset
Database1.pyproj # Project configuration file




---

## 🚀 Usage
1. Run **`Database1.py`** to preprocess the raw data.  
2. Use **`Book27.csv`** as the dataset for subsequent modules.  
3. Execute `module2`–`module8` to train and evaluate models.  
4. Review printed metrics to identify the best-performing model.

---

## 📊 Results (Summary)
- **SMOTE + XGBoost / Stacking** → highest recall and AUC.  
- **Neural network** effective but sensitive to overfitting.  
- **Top predictors:** Health, socioeconomic factors, and marital status.  

---

## 🔮 Future Enhancements
- Add **cross-validation** and **hyperparameter optimization** (`GridSearchCV`).  
- Develop **multi-year longitudinal models**.  
- Integrate **explainable AI** dashboards using SHAP or LIME.

---

## 📜 License
Distributed under the [MIT License](LICENSE).

---

## 📚 References
- IPUMS NHIS Documentation  
- CDC National Health Interview Survey (NHIS)  
- scikit-learn & TensorFlow official guides  

---

> “Without data, you’re just another person with an opinion.”  
> — W. Edwards Deming



