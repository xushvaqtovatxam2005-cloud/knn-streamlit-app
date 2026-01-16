# bu yerga Streamlit kodini joylang
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# SAHIFA SOZLAMALARI
# ---------------------------------------
st.set_page_config(page_title="KNN Classification App", layout="wide")
st.title("💊 KNN Classification Web App")
st.write("CSV fayl yuklang va KNN modeli natijalarini ko‘ring")

# ---------------------------------------
# 1–2-QADAM: CSV yuklash
# ---------------------------------------
file = st.file_uploader("📂 CSV faylni yuklang", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # ---------------------------------------
    # 3-QADAM: Datasetni ko‘rish
    # ---------------------------------------
    st.subheader("📄 Dataset (birinchi 5 qator)")
    st.dataframe(df.head())

    st.subheader("📊 Dataset haqida ma’lumot")
    st.write(df.describe())

    # ---------------------------------------
    # 4-QADAM: X va Y ajratish
    # ---------------------------------------
    median_charges = df['charges'].median()
    y = (df['charges'] > median_charges).astype(int)
    X = df.drop('charges', axis=1)

    X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

    # ---------------------------------------
    # 5-QADAM: Train / Test
    # ---------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------
    # 6-QADAM: StandardScaler
    # ---------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------------------
    # 7-QADAM: KNN modeli
    # ---------------------------------------
    k = st.slider("🔢 K qiymatini tanlang", 1, 15, 5)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # ---------------------------------------
    # 8-QADAM: Bashorat
    # ---------------------------------------
    y_pred = model.predict(X_test)

    # ---------------------------------------
    # 9-QADAM: Classification Report
    # ---------------------------------------
    st.subheader("📈 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # ---------------------------------------
    # 10-QADAM: Confusion Matrix
    # ---------------------------------------
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------------------------------
    # 11-QADAM: Precision / Recall / F1 grafik
    # ---------------------------------------
    st.subheader("📊 Precision – Recall – F1-score")

    scores = report_df.iloc[:-1][['precision', 'recall', 'f1-score']]
    fig2, ax2 = plt.subplots(figsize=(8,5))
    scores.plot(kind='bar', ax=ax2)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True)
    st.pyplot(fig2)

    # ---------------------------------------
    # 12-QADAM: Sinflar taqsimoti
    # ---------------------------------------
    st.subheader("📉 Sinflar taqsimoti")

    fig3, ax3 = plt.subplots()
    sns.countplot(x=y, ax=ax3)
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

else:
    st.info("👆 Iltimos, CSV fayl yuklang")
st.subheader("🧑‍⚕️ Yangi bemor ma’lumotini kiriting")

age = st.number_input("Yosh", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", value=25.0)
children = st.number_input("Bolalar soni", min_value=0, max_value=10, value=0)

sex = st.selectbox("Jins", ["male", "female"])
smoker = st.selectbox("Chekuvchi", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])


input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex": [sex],
    "smoker": [smoker],
    "region": [region]
})


input_data = pd.get_dummies(input_data)

# Train dagi ustunlar bilan tenglashtiramiz
input_data = input_data.reindex(columns=X.columns, fill_value=0)


input_scaled = scaler.transform(input_data)


if st.button("🔍 Natijani aniqlash"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("❗ Bemor KASAL (yuqori xavf)")
    else:
        st.success("✅ Bemor KASAL EMAS (past xavf)")


