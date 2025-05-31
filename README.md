# 🎓 Student Performance Prediction using Logistic Regression

## 📌 Project Title
**Student Performance Prediction using Logistic Regression**

---

## 📅 Week 2 – Mini Project (AI & ML)

This project builds a simple machine learning model to predict whether a student will **pass or fail** based on:

- Hours studied
- Attendance percentage

The model uses **Logistic Regression**, a supervised classification algorithm.

---

## 🎯 Objective

To:

- Understand and apply a basic ML classification algorithm
- Train and test a model on a small dataset
- Predict new outcomes
- Evaluate accuracy

This project is intended for beginners exploring core ML concepts.

---

## 📊 Dataset Description

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

> 📌 Note: The dataset contains 10 samples and is for learning/demo purposes only.

---

## 🛠️ Tools and Libraries Used

| Library            | Purpose                                    |
|--------------------|--------------------------------------------|
| `pandas`           | Data handling                              |
| `matplotlib.pyplot`| Data visualization                         |
| `sklearn`          | ML model training, testing, evaluation     |

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---

## 📈 Machine Learning Workflow

1. **Dataset Creation**  
   The dataset is hardcoded using Python dictionaries and converted into a pandas DataFrame.

2. **Data Visualization**  
   A scatter plot is created to visualize the data points and distinguish pass/fail students using colors.

3. **Data Splitting**  
   Features (`Hours_Studied`, `Attendance`) and labels (`Pass/Fail`) are separated and split into training and testing sets using `train_test_split()`.

4. **Model Training**  
   A Logistic Regression model is trained using the training data.

5. **Model Evaluation**  
   Predictions are made on the test set and the model's accuracy is computed using `accuracy_score()`.

6. **Prediction on New Data**  
   The trained model predicts the pass/fail outcome for new student data.

---

## 🧪 Results

### ✅ Model Accuracy  
The model achieves high accuracy on the test set, typically close to 100% due to the small dataset size.

### ✅ Example Prediction

```python
new_data = pd.DataFrame({
    'Hours_Studied': [2, 5, 6],
    'Attendance': [55, 80, 90]
})
predictions = model.predict(new_data)
print(predictions)
```

**Output:**

| Hours_Studied | Attendance | Predicted_Pass_Fail |
|---------------|------------|---------------------|
| 2             | 55         | 0                   |
| 5             | 80         | 1                   |
| 6             | 90         | 1                   |

---

## 📁 Project Files

| File Name                                 | Description                         |
|-------------------------------------------|-------------------------------------|
| `student_performance_logistic_regression.py` | Python script with full code       |
| `README.md`                               | Project documentation              |

---

## 🚀 How to Run This Project

1. Open your preferred Python environment (Google Colab, Jupyter Notebook, or local IDE).
2. Install required libraries if needed:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
3. Run the script or notebook cells step-by-step.
4. Modify input data as needed to make new predictions.

---

## ⚠️ Important Notes

- This project uses a **very small dataset** and is meant purely for educational purposes.
- Logistic Regression works well here due to the simple, binary classification problem.
- Accuracy may be misleadingly high given the dataset size; do not use this model for real-world decisions.

---

## 👤 Author

- **Name:** Fahim Akthar B  
- **Institution:** Crescent Institute of Science & Technology  
- **Course:** AI & ML – Week 2 Mini Project

31/05/25
