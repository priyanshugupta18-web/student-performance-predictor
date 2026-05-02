import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Predictor", layout="centered")

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

age = st.number_input("Age (in years)", min_value=15, max_value=22, value=17)

studytime = st.number_input(
    "Study Time (1:<2h, 2=2–5h, 3=5–10h, 4>10h)",
    min_value=1, max_value=4, value=2
)

failures = st.number_input(
    "Past Failures (number of previous failures)",
    min_value=0, max_value=3, value=0
)

absences = st.number_input(
    "Absences (number of school absences)",
    min_value=0, max_value=50, value=5
)

G1 = st.number_input(
    "G1 (first exam grade out of 20)",
    min_value=0, max_value=20, value=10
)

G2 = st.number_input(
    "G2 (second exam grade out of 20)",
    min_value=0, max_value=20, value=10
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
