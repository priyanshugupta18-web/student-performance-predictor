import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.title("🎓 Student Performance Predictor")

df = pd.read_csv("student_data.csv")
df.drop_duplicates(inplace=True)

features = ['sex', 'age', 'studytime', 'failures', 'absences', 'G1', 'G2']
df_model = df[features + ['G3']].copy()

df_model['result'] = (df_model['G3'] >= 10).astype(int)
df_model.drop('G3', axis=1, inplace=True)

df_model['sex'] = df_model['sex'].map({'F': 0, 'M': 1})

X = df_model.drop('result', axis=1)
y = df_model['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("📈 Model Accuracy")
st.write(round(accuracy_score(y_test, y_pred), 2))

st.subheader("📥 Enter Student Details")

sex = st.selectbox("Gender", ["Male", "Female"])

age = st.slider("Age (Student age in years)", 15, 22, 17)

studytime = st.slider(
    "Study Time (Weekly study hours level: 1 <2h, 2=2–5h, 3=5–10h, 4 >10h)",
    1, 4, 2
)

failures = st.slider(
    "Past Failures (Number of past class failures)",
    0, 3, 0
)

absences = st.slider(
    "Absences (Number of school absences)",
    0, 50, 5
)

G1 = st.slider(
    "G1 (First period grade out of 20)",
    0, 20, 10
)

G2 = st.slider(
    "G2 (Second period grade out of 20)",
    0, 20, 10
)

sex_val = 1 if sex == "Male" else 0

if st.button("Predict"):

    input_data = [[sex_val, age, studytime, failures, absences, G1, G2]]
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")

    st.subheader("📊 Your Input Visualization")

    labels = ['Study Time', 'Failures', 'Absences', 'G1', 'G2']
    values = [studytime, failures, absences, G1, G2]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)
