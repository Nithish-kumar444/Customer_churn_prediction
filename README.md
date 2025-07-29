# Customer_churn_prediction

This project aims to build a machine learning model that predicts whether a customer is likely to churn based on historical behavior and attributes. Churn prediction helps businesses proactively reduce customer loss.

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

---

## 📦 Libraries and Tools

- numpy  
- pandas  
- matplotlib  
- seaborn  
- pickle  
- scikit-learn  
- xgboost  
- imbalanced-learn (SMOTE, RandomUnderSampler)

---

## 📝 Features

- Data loading and preprocessing  
- Label Encoding for categorical variables  
- Handling class imbalance using:  
  - SMOTE (Over-sampling)  
  - Random Under Sampler  
- Model training and evaluation  
- Performance metrics:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-score
  - ROC-AUC Score  
- Cross-validation  
- Model saving with `pickle`

---


---

## 🧪 Evaluation Metrics

- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, F1 Score  
- ROC AUC Score  
- Classification Report  

Cross-validation is used to ensure model robustness.

---

## ⚖️ Handling Imbalanced Data

- **SMOTE** (Synthetic Minority Over-sampling Technique)  
- **Random Under Sampling**

These methods help balance the number of churned vs. non-churned customers.

---

## 💾 Model Saving

  ```python
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(model, f)
```
## 🚀 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/customer_churn_prediction.git
cd customer_churn_prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run:

bash
Copy
Edit
python churn_prediction.py
✅ Requirements
Sample requirements.txt:

nginx
Copy
Edit
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
📌 Notes
Ensure the dataset has a target column like Churn or Exited.

Adjust column names and preprocessing steps accordingly.

📉 Visualizations
## correlation heat map
<img width="1920" height="1080" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/e2efe373-aacf-4eda-a77a-4aaddd330ecd" />
## histogram graph 
<img width="1920" height="1080" alt="Screenshot (19)" src="https://github.com/user-attachments/assets/ac000c3d-f22b-4439-87b8-1b193cdeae82" />
## box_plot graph
<img width="1920" height="1080" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/8bd7d927-190b-4d9f-9cea-1b3f26000a01" />

## dashboard for the customer churn prediction

This project provides an interactive dashboard for **Customer Churn Prediction** using **Streamlit** and **Pyngrok**, allowing easy web-based access even from a local machine. The dashboard allows users to input customer data and get churn predictions in real time.

---

## 🧰 Tech Stack

- **Streamlit** – Web app framework for data apps  
- **Pyngrok** – To expose Streamlit app on a public URL  
- **Scikit-learn / XGBoost** – For model inference  
- **Pandas, Numpy** – Data handling  
- **Pickle** – Model loading  

---

## 📁 Project Structure

customer_churn_dashboard/
│
├── models/
│ └── best_model.pkl # Trained ML model
│
├── dashboard.py # Streamlit app
├── README.md # This file
└── requirements.txt # Dependencies

---

## 🚀 Features

- Upload or enter customer data
- Predict whether the customer is likely to churn
- Display prediction result with probability
- Interactive UI using Streamlit widgets
- Expose the dashboard to the internet using Pyngrok

---
## Snapshot of the dashboard
<img width="1920" height="1080" alt="Screenshot (25)" src="https://github.com/user-attachments/assets/9a696b37-07ec-47a2-91e0-1eee84decdd4" />
<img width="1920" height="1080" alt="Screenshot (27)" src="https://github.com/user-attachments/assets/40307ad6-e689-4ce8-b293-49375c39620c" />

## 💻 How to Run the Dashboard

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
