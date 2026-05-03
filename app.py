import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Student Grade Predictor", layout="centered")

st.title("🎓 Student Final Grade Predictor")

df = pd.read_csv("student_data.csv")
df.drop_duplicates(inplace=True)

features = ['sex', 'age', 'studytime', 'failures', 'absences', 'G1', 'G2']
df_model = df[features + ['G3']].copy()

df_model['sex'] = df_model['sex'].map({'F': 0, 'M': 1})

X = df_model.drop('G3', axis=1)
y = df_model['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("📈 Model Performance")
st.write("R2 Score:", round(r2_score(y_test, y_pred), 2))
st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

st.subheader("📥 Enter Student Details")

sex = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (in years)", 15, 22, 17)

studytime = st.number_input(
    "Study Time (1:<2h, 2=2–5h, 3=5–10h, 4>10h)",
    1, 4, 2
)

failures = st.number_input(
    "Past Failures (number of previous failures)",
    0, 3, 0
)

absences = st.number_input(
    "Absences (number of school absences)",
    0, 50, 5
)

G1 = st.number_input(
    "G1 (first exam grade out of 20)",
    0, 20, 10
)

G2 = st.number_input(
    "G2 (second exam grade out of 20)",
    0, 20, 10
)

sex_val = 1 if sex == "Male" else 0

if st.button("Predict Final Grade"):

    input_df = pd.DataFrame(
        [[sex_val, age, studytime, failures, absences, G1, G2]],
        columns=['sex', 'age', 'studytime', 'failures', 'absences', 'G1', 'G2']
    )

    prediction = model.predict(input_df)[0]

    st.success(f"🎯 Predicted Final Grade: {round(prediction, 2)} / 20")

    if prediction >= 10:
        st.success("✅ Likely to PASS")
    else:
        st.error("❌ Likely to FAIL")

    st.subheader("📊 Your Input Visualization")

    labels = ['Study Time', 'Failures', 'Absences', 'G1', 'G2']
    values = [studytime, failures, absences, G1, G2]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)
