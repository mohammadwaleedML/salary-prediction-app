import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="wide"
)

model = joblib.load(os.path.join(BASE_DIR, "model", "salary_prediction_model.pkl"))

gender_encoder = joblib.load(os.path.join(BASE_DIR, "model", "gender_encoder.pkl"))

education_encoder = joblib.load(os.path.join(BASE_DIR, "model", "education_encoder.pkl"))

skills_encoder = joblib.load(os.path.join(BASE_DIR, "model", "skills_encoder.pkl"))

city_encoder = joblib.load(os.path.join(BASE_DIR, "model", "city_encoder.pkl"))

company_encoder = joblib.load(os.path.join(BASE_DIR, "model", "company_encoder.pkl"))

scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "salary_dataset_500.csv"))

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Salary Prediction",
        "Dataset Overview",
        "Analytics"
    ]
)


if page == "Salary Prediction":

    st.title("💰 Salary Prediction App")

    st.write("Predict salary using Machine Learning.")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 60, 25)

        gender = st.radio("Gender",["Male", "Female"])

        experience = st.slider("Experience",0,40,2)

        education = st.selectbox("Education",["Intermediate","Bachelors","Masters","PhD"])

    with col2:

        skills = st.selectbox("Skills",["Python","Java","SQL","Data Analysis","Machine Learning","Excel","Web Development"])

        city = st.selectbox("City",["Karachi","Lahore","Islamabad","Faisalabad","Peshawar","Quetta"])

        company_size = st.selectbox("Company Size",["Startup","Small","Medium","Large"])

    if st.button("Predict Salary"):

        gender_encoded = gender_encoder.transform([gender])[0]

        education_encoded = education_encoder.transform([education])[0]

        skills_encoded = skills_encoder.transform([skills])[0]

        city_encoded = city_encoder.transform([city])[0]

        company_encoded = company_encoder.transform([company_size])[0]

        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender_encoded],
            "experience": [experience],
            "education": [education_encoded],
            "skills": [skills_encoded],
            "city": [city_encoded],
            "company_size": [company_encoded]
                })

        input_data[["age", "experience"]] = scaler.transform(input_data[["age", "experience"]])

        prediction = model.predict(input_data)

        st.success(f"💰 Predicted Salary = Rs {prediction[0]:,.0f}")


elif page == "Dataset Overview":

    st.title("📋 Dataset Overview")

    st.dataframe(df)

    st.write("Shape of Dataset:", df.shape)

    st.write("Columns:", df.columns.tolist())

elif page == "Analytics":

    st.title("📊 Salary Analytics")

    # Average Salary
    avg_salary = int(df["salary"].mean())

    max_salary = int(df["salary"].max())

    min_salary = int(df["salary"].min())

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Salary", f"Rs {avg_salary}")

    col2.metric("Maximum Salary", f"Rs {max_salary}")

    col3.metric("Minimum Salary", f"Rs {min_salary}")

    st.subheader("Salary Distribution")


    fig, ax = plt.subplots()

    ax.hist(df["salary"], bins=20)

    st.pyplot(fig)

    
# Experience vs Salary
    

    st.subheader("Experience vs Salary")

    fig2, ax2 = plt.subplots()

    ax2.scatter(
        df["experience"],
        df["salary"]
    )

    ax2.set_xlabel("Experience")

    ax2.set_ylabel("Salary")

    st.pyplot(fig2)

# City Wise Salary

    st.subheader("City Wise Average Salary")

    city_salary = df.groupby("city")["salary"].mean()

    st.bar_chart(city_salary)
