import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# -------------------- UI CONFIG --------------------
st.set_page_config(page_title="Music Personality Analyzer", layout="centered")

st.title("🎵 Music Personality Analyzer")
st.markdown("Find your personality based on your music taste")

# -------------------- LOAD DATA --------------------
df = pd.read_csv("spotify.csv")

# -------------------- FEATURE SELECTION --------------------
features = df[['energy', 'danceability', 'valence', 'tempo', 'loudness']]
features = features.dropna()

# -------------------- SCALING --------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# -------------------- K-MEANS --------------------
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

features['cluster'] = clusters

# -------------------- CLUSTER LABELING --------------------
cluster_map = {
    0: 'Chill',
    1: 'Sad',
    2: 'Energetic',
    3: 'Happy',
    4: 'Aggressive'
}

features['music_type'] = features['cluster'].map(cluster_map)

# -------------------- MUSIC → PERSONALITY --------------------
def get_personality(music_type):
    if music_type == "Energetic":
        return "Energetic"
    elif music_type == "Chill":
        return "Chill"
    elif music_type == "Happy":
        return "Happy"
    elif music_type == "Sad":
        return "Sad"
    else:
        return "Aggressive"

features['personality'] = features['music_type'].apply(get_personality)

# -------------------- MODEL DATA --------------------
X = scaled_data
y = features['personality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- TRAIN MODELS --------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

# -------------------- MODEL ACCURACY --------------------
st.subheader("📊 Model Accuracy")

knn_acc = accuracy_score(y_test, knn.predict(X_test))
nb_acc = accuracy_score(y_test, nb.predict(X_test))

st.write(f"KNN Accuracy: {knn_acc:.2f}")
st.write(f"Naive Bayes Accuracy: {nb_acc:.2f}")

# -------------------- USER INPUT --------------------
st.subheader("🎚️ Enter Your Music Preferences")

energy = st.slider("Energy", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)

# -------------------- PERSONALITY MAP --------------------
personality_map = {
    "Energetic": ("The High-Energy Type", "You like to stay active and enjoy fast-paced vibes."),
    "Chill": ("The Relaxed Soul", "You prefer a calm environment and peaceful music."),
    "Happy": ("The Positive Spirit", "You have an optimistic outlook and love feel-good music."),
    "Sad": ("The Deep Thinker", "You connect deeply with emotions and meaningful lyrics."),
    "Aggressive": ("The Intense Personality", "You are bold and enjoy music with power and impact.")
}

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Personality"):

    user_input = [[energy, danceability, valence, tempo, loudness]]

    user_scaled = scaler.transform(user_input)

    predicted_personality = knn.predict(user_scaled)[0]

    personality_data = personality_map.get(
        predicted_personality,
        ("Music Lover", "You enjoy a wide variety of musical styles.")
    )

    label, description = personality_data

    st.success(f"🎯 Personality: {label}")
    st.write(description)

    # -------------------- VISUAL --------------------
    st.subheader("📊 Your Music Profile")

    import matplotlib.pyplot as plt

    labels = ['Energy', 'Danceability', 'Valence']
    values = [energy, danceability, valence]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)
