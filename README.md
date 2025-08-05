# Heart Disease Prediction Using Machine Learning

This project aims to predict the presence of heart disease in patients using machine learning classification models. It is a part of an academic machine learning task where we preprocess the data, engineer features, apply multiple models, and evaluate their performance on various metrics.

## 👨‍💻 Author

**Nasir Sharif**

---

## 📂 Project Structure

- `heart.csv`: Dataset used for model training and testing.
- `Heart_Disease_Prediction.ipynb`: Main notebook with data analysis, modeling, and evaluation.
- `README.md`: Project documentation.

---

## 📊 Dataset

The dataset consists of 303 rows and 14 features related to patient medical records such as:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope of ST
- Number of Major Vessels
- Thalassemia
- Target (0 = No Disease, 1 = Disease Present)

---

## 🚀 Key Steps

### ✅ Part 1: Data Preprocessing & Modeling
- Handled missing values with mean imputation.
- Scaled data using StandardScaler.
- Applied **Polynomial Features**.
- Feature selection using **SelectKBest**.
- Trained models: Logistic Regression, Random Forest, SVM, Gradient Boosting, etc.
- Selected **Logistic Regression** as best model based on performance.

### 🔬 Part 2: Feature Engineering
- Applied:
  - **Polynomial Features** to introduce interaction terms.
  - **KMeans Clustering Labels** to assign group labels.
- Retrained and improved accuracy to **98.53%** and **F1 Score to 0.9852**.
- Demonstrated how engineered features can boost performance.

### 📈 Part 3: Evaluation Metric Justification

#### Most Suitable Metric: **Recall** (and F1 Score)

- In healthcare, **False Negatives** (predicting no disease when disease is present) can be deadly.
- High **Recall** ensures that we identify the maximum number of patients with heart disease.
- F1 Score balances both Recall and Precision for better evaluation.

---

## 🧠 Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- K-Nearest Neighbors
- Decision Tree

---

## 📉 Performance Summary

| Metric | Score |
|--------|-------|
| Accuracy | 0.985 |
| F1 Score | 0.985 |
| ROC AUC  | 0.976 |

---

## 📌 Requirements

- Python 3.8+
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- NumPy

---

## 📁 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Nasir-Sharif/Google-Colab-File-Cardio-Net.git
