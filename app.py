import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("Student Placement Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload Placement CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Required columns (based on your Kaggle dataset)
    required_columns = [
        'Internships',
        'Projects',
        'Coding_Skills',
        'Communication_Skills',
        'Aptitude_Test_Score',
        'Soft_Skills_Rating',
        'Placement_Status'
    ]

    if all(col in df.columns for col in required_columns):

        # Features and Target
        X = df[
            ['Internships',
             'Projects',
             'Coding_Skills',
             'Communication_Skills',
             'Aptitude_Test_Score',
             'Soft_Skills_Rating']
        ]

        y = df['Placement_Status']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Training
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        st.success(f"Model Accuracy: {round(accuracy*100,2)}%")

        st.subheader("🔮 Enter Student Details")

        internships = st.number_input("Internships", 0, 10, 1)
        projects = st.number_input("Projects", 0, 10, 2)
        coding = st.slider("Coding Skills", 1, 10, 5)
        communication = st.slider("Communication Skills", 1, 10, 5)
        aptitude = st.slider("Aptitude Test Score", 0, 100, 50)
        softskills = st.slider("Soft Skills Rating", 1, 10, 5)

        if st.button("Predict Placement"):

            input_data = np.array([[internships, projects, coding,
                                    communication, aptitude, softskills]])

            # IMPORTANT: scale input
            input_data = scaler.transform(input_data)

            probability = model.predict_proba(input_data)[0][1]

            st.info(f"Placement Probability: {round(probability*100,2)}%")

            if probability >= 0.5:
                st.success("✅ High Chance of Placement")
            else:
                st.error("❌ Low Chance of Placement")

    else:
        st.error("❌ Dataset columns do not match required format.")