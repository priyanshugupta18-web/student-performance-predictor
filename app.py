import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Student Grade Predictor", layout="centered")
st.title("🎓 Student Final Grade Predictor")

# --- Model Logic ---
@st.cache_data
def get_trained_model():
    df = pd.read_csv("student_data.csv")
    df.drop_duplicates(inplace=True)
    features = ['sex', 'age', 'studytime', 'failures', 'absences', 'G1', 'G2']
    df_model = df[features + ['G3']].copy()
    df_model['sex'] = df_model['sex'].map({'F': 0, 'M': 1})
    X = df_model.drop('G3', axis=1)
    y = df_model['G3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, y_test, model.predict(X_test)

model, y_test, y_pred = get_trained_model()

st.subheader("📈 Model Performance")
st.write(f"R2 Score: {round(r2_score(y_test, y_pred), 2)} | MAE: {round(mean_absolute_error(y_test, y_pred), 2)}")

st.divider()

# --- Progressive Input Logic ---
if 'step' not in st.session_state:
    st.session_state.step = 1

# Step 1 to 6 (Same as before)
if st.session_state.step >= 1:
    sex = st.selectbox("1. Gender", ["Male", "Female"], key="s1")
    if st.session_state.step == 1:
        if st.button("Next ➡️", key="b1"): 
            st.session_state.step = 2
            st.rerun()

if st.session_state.step >= 2:
    age = st.number_input("2. Age (15-22)", 15, 22, 17, key="s2")
    if st.session_state.step == 2:
        if st.button("Next ➡️", key="b2"):
            st.session_state.step = 3
            st.rerun()

if st.session_state.step >= 3:
    study_labels = {"<2 hours": 1, "2-5 hours": 2, "5-10 hours": 3, ">10 hours": 4}
    study_choice = st.selectbox("3. Weekly Study Time", list(study_labels.keys()), key="s3")
    studytime = study_labels[study_choice]
    if st.session_state.step == 3:
        if st.button("Next ➡️", key="b3"):
            st.session_state.step = 4
            st.rerun()

if st.session_state.step >= 4:
    failures = st.number_input("4. Past Failures (0-3)", 0, 3, 0, key="s4")
    if st.session_state.step == 4:
        if st.button("Next ➡️", key="b4"):
            st.session_state.step = 5
            st.rerun()

if st.session_state.step >= 5:
    absences = st.number_input("5. Total Absences", 0, 100, 5, key="s5")
    if st.session_state.step == 5:
        if st.button("Next ➡️", key="b5"):
            st.session_state.step = 6
            st.rerun()

if st.session_state.step >= 6:
    G1 = st.number_input("6. G1 Score (0-20)", 0, 20, 10, key="s6")
    if st.session_state.step == 6:
        if st.button("Next ➡️", key="b6"):
            st.session_state.step = 7
            st.rerun()

# Step 7: G2 & Full Logic
if st.session_state.step >= 7:
    G2 = st.number_input("7. G2 Score (0-20)", 0, 20, 10, key="s7")
    
    if st.button("Predict Final Grade 🎯"):
        sex_val = 1 if sex == "Male" else 0
        input_df = pd.DataFrame(
            [[sex_val, age, studytime, failures, absences, G1, G2]],
            columns=['sex', 'age', 'studytime', 'failures', 'absences', 'G1', 'G2']
        )
        prediction = model.predict(input_df)[0]
        
        st.divider()
        st.success(f"### Predicted Final Grade: {round(prediction, 2)} / 20")
        
        # --- PASS/FAIL Logic Wapas Add Kiya ---
        if prediction >= 10:
            st.balloons()
            st.success("✅ Likely to PASS")
        else:
            st.error("❌ Likely to FAIL")

        # --- Graph Section ---
        st.subheader("📊 Your Input Visualization")
        labels = ['Study Time', 'Failures', 'Absences', 'G1', 'G2']
        values = [studytime, failures, absences, G1, G2]
        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['#4CAF50' if prediction >= 10 else '#FF5252'])
        st.pyplot(fig)

    if st.button("Reset Form 🔄"):
        st.session_state.step = 1
        st.rerun()