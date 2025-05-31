# 🎓 Student Performance Prediction using Logistic Regression

---

## 📌 Project Title
**Student Performance Prediction using Logistic Regression**

---

## 📅 Week 2 – Mini Project (AI & ML)

This project aims to build a simple machine learning model to predict whether a student will **pass or fail** based on:
- The number of hours they studied
- Their attendance percentage

The model is built using **Logistic Regression**, a supervised learning classification algorithm.

---

## 🎯 Objective

The main objectives of this project are:

- To understand and implement logistic regression for binary classification
- To learn the basic ML workflow: data creation, visualization, training, testing, and prediction
- To evaluate model performance on test data
- To make predictions on new input values

This project is designed for beginners and focuses on gaining hands-on experience with a simple, interpretable algorithm.

---

## 📊 Dataset Description

The dataset is manually created for learning purposes. It consists of the following features:

| Feature Name     | Description                            |
|------------------|----------------------------------------|
| Hours_Studied    | Number of hours a student studied      |
| Attendance       | Student's attendance percentage        |
| Pass/Fail        | Target label: 1 = Pass, 0 = Fail       |

### 🔢 Sample Data

| Hours_Studied | Attendance | Pass/Fail |
|---------------|------------|-----------|
| 5             | 85         | 1         |
| 2             | 60         | 0         |
| 4             | 75         | 1         |
| 1             | 50         | 0         |

> Note: The dataset contains 10 such samples and is designed only for learning and experimentation.

---

## 🛠️ Tools and Libraries Used

The project was developed in Python using the following libraries:

| Library          | Purpose                                     |
|------------------|---------------------------------------------|
| pandas           | Data creation and handling                  |
| matplotlib       | Data visualization                          |
| sklearn (scikit-learn) | ML model creation, training, testing, evaluation |

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
📈 Machine Learning Workflow
1. Dataset Creation
The dataset is created using Python dictionaries and converted into a DataFrame using pandas.

2. Data Visualization
A scatter plot is used to visualize the data and class distribution (pass/fail) using color mapping.

3. Data Splitting
Features (Hours_Studied, Attendance) and label (Pass/Fail) are separated and split into training and test sets using train_test_split().

4. Model Training
A logistic regression model is trained on the training dataset.

5. Model Testing & Evaluation
The model's predictions on the test set are compared with actual labels to calculate accuracy.

6. Prediction on New Data
The trained model is used to make predictions for new student data points.

🧪 Results
✅ Model Accuracy
The model achieved an accuracy of 100% on the test data (this may vary based on the train/test split).

✅ Example Prediction
python
Copy
Edit
new_data = pd.DataFrame({
    'Hours_Studied': [2, 5, 6],
    'Attendance': [55, 80, 90]
})
predictions = model.predict(new_data)
Output:

nginx
Copy
Edit
   Hours_Studied  Attendance  Predicted_Pass_Fail
0            2.0          55                    0
1            5.0          80                    1
2            6.0          90                    1
📁 Project Files
File Name	Description
student_performance_logistic_regression.py	Python script with the full ML workflow
README.md	Project documentation

🚀 How to Run This Project
Open your Python environment (Jupyter, Google Colab, or any IDE)

Install required libraries (if not already installed):

bash
Copy
Edit
pip install pandas matplotlib scikit-learn
Run the .py or .ipynb file

View plots, accuracy, and make new predictions

📌 Important Notes
The dataset is extremely small and not suitable for production.

This is a concept learning project for beginners.

Logistic regression is ideal for learning basic classification in ML.

👤 Author
Name: Fahim Akthar B

Institution: Crescent Institute of Science & Technology

Course: AI & ML – Week 2 Mini Project
