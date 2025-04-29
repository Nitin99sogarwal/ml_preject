import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

st.header("📚 Student Dropout Risk Prediction 📉")
st.write("📊 *Predictive Model Built on Student Data*")

try:
    df = pd.read_csv("data.csv")

    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    if 'Test_Scores' in df.columns:
        df = df.rename(columns={'Test_Scores': 'Test_Scores_median'})

    # st.dataframe(df.head())  
except FileNotFoundError:
    st.error("❌ Dataset file 'data.csv' not found. Please upload the file.")
    st.stop()

st.write("🌟 *Enter the following student data for prediction:*")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Select Gender", options=["Female", "Male"])
    Gender = 0 if Gender == "Female" else 1

with col2:
    Parental_Education = st.selectbox("Select Parental Education Level", options=["Uneducated", "Primary", "Secondary", "Tertiary"])
    Parental_Education = {"Uneducated": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}[Parental_Education]

col3, col4 = st.columns(2)
with col3:
    Attendance_Rate = st.number_input("Enter Attendance Rate (%)", min_value=0.0, max_value=100.0, step=0.1)

with col4:
    SocioEconomic_Status = st.selectbox("Select Socioeconomic Status", options=["Low", "Medium", "High"])
    SocioEconomic_Status = {"Low": 0, "Medium": 1, "High": 2}[SocioEconomic_Status]

col5, col6 = st.columns(2)
with col5:
    Engagement_Score = st.number_input("Enter Engagement Score (0 to 100)", min_value=0.0, max_value=100.0, step=0.1)

with col6:
    Test_Scores_median = st.number_input("Enter Test Scores Median (0 to 100)", min_value=0.0, max_value=100.0, step=0.1)

input_data = [
    [Gender, Parental_Education, Attendance_Rate, SocioEconomic_Status, Engagement_Score, Test_Scores_median]
]


try:
    model = joblib.load('decision.pkl')
except FileNotFoundError:
    st.error("❌ Model not found. Please upload the model.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

if st.button("🔍 Predict Dropout Risk"):
    try:
        input_data = [[int(Gender), int(Parental_Education), float(Attendance_Rate), int(SocioEconomic_Status), float(Engagement_Score), float(Test_Scores_median)]]
        
        prediction = model.predict(input_data)
        # st.markdown(prediction[0])
        if prediction[0] > 1:
            st.markdown("<h3 style='color: red;'>The student is at risk of dropping out. 🚨</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>The student is not at risk of dropping out. ✅</h3>", unsafe_allow_html=True)
        
        st.write("💾 *Given Input:*")
        input_df = pd.DataFrame(input_data, columns=[
            "Gender", "Parental Education", "Attendance Rate", 
            "Socioeconomic Status", "Engagement Score", "Test Scores Median"
        ])
        st.dataframe(input_df)
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")