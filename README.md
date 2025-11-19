
# 📱 Expresso Churn Prediction Project
A machine learning project that predicts customer churn for Expresso, a telecommunications company operating in Mauritania and Senegal. This project was built as part of a data science checkpoint and includes data exploration, preprocessing, model training, and a Streamlit web application for live predictions.

---

## 📊 **Project Overview**
Customer churn is one of the most important business metrics for telecom companies. The goal of this project is to build a classification model that predicts the likelihood of a customer leaving (churning) based on their usage behavior and account information.

The project includes:

- Data cleaning  
- Exploratory data analysis  
- Feature preprocessing  
- Machine learning model training  
- Saving the trained model  
- Deploying a local Streamlit prediction app  

---

## 📁 **Dataset Information**
The dataset comes from the **Expresso Churn Prediction Challenge** hosted on **Zindi**.

It contains:
- **2.5 million clients**
- **15+ behavior variables**
- **Binary churn target** (0 = No churn, 1 = Churn)

Example features may include:
- Customer activity days  
- Number of SMS  
- Data volume  
- Voice usage  
- ARPU (Average Revenue Per User)

---

## 🛠️ **Technologies & Libraries Used**
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Pandas Profiling**
- **Streamlit**
- **Joblib**

---

## 🧹 **Data Preparation Steps**
1. Loaded the dataset and explored structure  
2. Generated a **pandas profiling report** for insights  
3. Handled missing values  
4. Removed duplicated rows  
5. Handled outliers using IQR  
6. Encoded categorical features using `LabelEncoder`  
7. Split dataset into train/test subsets  

---

## 🤖 **Machine Learning Model**
A **Random Forest Classifier** was used because it handles:
- Large datasets  
- Nonlinear patterns  
- Mixed feature types  

Model evaluation included:
- Accuracy score  
- Classification report  
- Confusion matrix (optional)

The final trained model and feature list were saved using:

```

churn_model.pkl
model_features.pkl

````

---

## 🌐 **Streamlit Application**
A simple and interactive Streamlit app was built to allow users to input customer information and receive a churn prediction.

### **Running the app:**
```bash
streamlit run app.py
````

The app:

* Loads the trained model
* Accepts input values for all model features
* Predicts churn probability
* Displays whether the customer is likely to churn

---

## 📂 **Project Structure**

```
📁 Expresso-Churn-Prediction
│── app.py                  # Streamlit app
│── churn_model.pkl         # Trained model
│── model_features.pkl      # Feature columns
│── notebook.ipynb          # Data exploration & modeling (optional)
│── requirements.txt        # Dependencies list
│── README.md               # Project documentation
```

---

## ▶️ **How to Run the Project**

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/expresso-churn.git
cd expresso-churn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🚀 **Future Improvements**

* Add feature engineering
* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Deploy the Streamlit app online (Streamlit Cloud)
* Add SHAP feature importance explanations

---

## 👩🏽‍💻 **Author**

**Nicole Mugo**
Data Science & Machine Learning Enthusiast

---

## ⭐ **If you like this project, give it a star on GitHub!**
