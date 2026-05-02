# 🎵 Music Personality Analyzer

## 📌 Overview

This project predicts a user's personality based on their music preferences.
It uses Spotify song features like energy, danceability, and valence to analyze patterns in music and classify personality types.

---

## ⚙️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib

---

## 🧠 Methodology

1. **Data Collection**

   * Used a Spotify dataset containing audio features of songs

2. **Feature Selection**

   * Selected features: energy, danceability, valence, tempo, loudness

3. **Clustering (K-Means)**

   * Grouped songs into 5 clusters
   * Labeled clusters as: Chill, Sad, Energetic, Happy, Aggressive

4. **Label Creation**

   * Converted music types into personality labels

5. **Classification Models**

   * K-Nearest Neighbors (KNN)
   * Naive Bayes

6. **Prediction**

   * User inputs music preferences
   * Model predicts personality

---

## 🎯 Features

* Music clustering using K-Means
* Personality prediction using KNN and Naive Bayes
* Interactive UI using Streamlit
* Visualization of user preferences

---

## ▶️ How to Run

1. Install dependencies:

```
pip install pandas scikit-learn streamlit matplotlib
```

2. Run the app:

```
streamlit run app.py
```

---

## 📊 Example Output

* Personality Type: The High-Energy Type
* Description: You like to stay active and enjoy fast-paced vibes

---

## ⚠️ Note

This project is a simplified model and does not represent exact psychological analysis.

---

## 👨‍💻 Author

Priyanshu Gupta
