# 🩺 Diabetes Prediction Model

## 📌 Project Overview
Diabetes is one of the most common chronic diseases worldwide.  
This project applies **machine learning** to predict whether a patient has diabetes based on health-related features.  

The workflow:
- Start with **Logistic Regression** as a **baseline** (simple & interpretable).  
- Move to **LightGBM with Optuna tuning** for the **final production-ready model**.  

The emphasis is on building a model that is **generalized**, not overfitting or underfitting, while being accurate enough for healthcare applications.

---

## 📊 Dataset
- **Source**: PIMA Indian Diabetes Dataset  
- **Samples**: 768  
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age  
- **Target**: `Outcome` → `1` (diabetic), `0` (non-diabetic)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handled missing/invalid values (e.g., insulin = 0 replaced by imputation).  
- Scaled features for Logistic Regression.  
- LightGBM handled raw features directly (tree-based model).  

### 2. Baseline Model → Logistic Regression
- Provides an **interpretable starting point**.  
- Useful for quick benchmarking.  
- Threshold tuning improved recall and F1 score.  

### 3. Final Model → LightGBM with Optuna
- LightGBM (gradient boosting framework) chosen for:
  - High performance on tabular data  
  - Handling missing values & non-linear relationships  
- **Optuna** used for hyperparameter optimization:
  - Automated search for best parameters (learning rate, depth, num_leaves, etc.)  
  - Bayesian optimization → more efficient than GridSearchCV  

---

## 📈 Results

### Logistic Regression (Baseline)
| Metric     | Score |
|------------|-------|
| Accuracy   | ~0.79 |
| Precision  | ~0.67 |
| Recall     | ~0.81 |
| F1 Score   | ~0.74 |

### LightGBM + Optuna (Final Model)
| Metric     | Score |
|------------|-------|
| Accuracy   | **~0.85** |
| Precision  | **~0.76** |
| Recall     | **~0.82** |
| F1 Score   | **~0.79** |
| CV Accuracy| **~0.83 ± small variance** |

---

## 🔎 Interpretation
- **Logistic Regression** → good baseline, interpretable, but limited in capturing complex patterns.  
- **LightGBM (Final)** → higher performance, robust, and generalizes better.  
- **Generalization Check**: CV performance is close to test accuracy → not overfitting.  
- **Healthcare Relevance**:  
  - High **recall** ensures most diabetic patients are detected.  
  - Balanced precision avoids excessive false positives.  

---

## 🚀 Conclusion
- **Baseline Logistic Regression** proved the problem was solvable (~0.79 acc).  
- **Optuna-tuned LightGBM** achieved the best balance (~0.85 acc, ~0.79 F1).  
- The final model is **generalized, reliable, and healthcare-usable**.  

---

## 📂 Repository Structure
├── PIMA/
│   ├── .ipynb_checkpoints/
│   ├── app.py
│   ├── diabetes-predictor.ipynb
│   ├── diabetes.csv
│   ├── model.pkl
│   ├── README.md