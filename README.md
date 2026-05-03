# 🎓 Student Performance Predictor

## 📌 Overview

This project predicts a student’s academic performance using Machine Learning.

Two approaches are implemented:

* **Classification (KNN)** → Predicts Pass/Fail
* **Regression (Random Forest)** → Predicts exact final grade (out of 20)

The project is deployed using Streamlit for an interactive user interface.

---

## 🚀 Features

* Predict Pass/Fail outcome
* Predict exact final grade
* Interactive web app using Streamlit
* Clean and simple UI
* User input visualization
* Model performance display

---

## 📊 Dataset

* Student Performance Dataset
* Key features used:

  * Gender
  * Age
  * Study Time
  * Past Failures
  * Absences
  * G1 (First exam grade)
  * G2 (Second exam grade)

---

## 🧠 Models Used

### 🔹 KNN Classifier

* Used for Pass/Fail prediction
* Based on similarity between students

### 🔹 Random Forest Regressor

* Used for predicting final grade (G3)
* More powerful and accurate for numeric prediction

---

## ⚙️ How It Works

1. Data is loaded and cleaned
2. Features are selected
3. Data is preprocessed and encoded
4. Models are trained
5. User inputs data through Streamlit UI
6. Model predicts result and displays output

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

* Classification:

  * Accuracy
* Regression:

  * MAE (Mean Absolute Error)
  * R² Score

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

---

## 🎯 Conclusion

This project demonstrates how machine learning can be used to analyze and predict student performance using both classification and regression techniques.

---

## 👨‍💻 Author

**Priyanshu Gupta**
