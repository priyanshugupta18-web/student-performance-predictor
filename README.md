# 🎓 Student Performance Predictor

## 📌 Overview

This project predicts whether a student will **pass or fail** based on academic and personal factors using Machine Learning.

It uses the **K-Nearest Neighbors (KNN)** algorithm and is deployed using **Streamlit** for an interactive user interface.

---

## 🚀 Features

* Predict student performance (Pass/Fail)
* Simple and interactive UI using Streamlit
* Uses real-world student data
* Clean and minimal design
* Displays model accuracy
* Visualizes user input

---

## 📊 Dataset

* Dataset: Student Performance Dataset
* Features used:

  * Gender
  * Age
  * Study Time (weekly study hours level)
  * Past Failures
  * Absences
  * G1 (First exam grade)
  * G2 (Second exam grade)

---

## 🧠 Model Used

* **K-Nearest Neighbors (KNN)**
* Reason:

  * Easy to understand
  * Works well with small datasets
  * Based on similarity between data points

---

## ⚙️ How It Works

1. Data is loaded and cleaned
2. Features are selected
3. Target variable (Pass/Fail) is created
4. Data is scaled using StandardScaler
5. KNN model is trained
6. User inputs data via Streamlit UI
7. Model predicts whether student will pass or fail

---

## 🖥️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

---

## 📈 Model Evaluation

* Metric used: **Accuracy**
* The model predicts based on similarity with other students

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

---

## 🎯 Conclusion

This project demonstrates how machine learning can be used to predict student performance and assist in academic analysis.

---

## 👨‍💻 Author

* Priyanshu Gupta
