#Libaries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#CSS style
st.markdown("""
    <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        h1, h3, h4 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


#Create Dataframe s for social media usage
s = pd.read_csv('social_media_usage.csv')

#Define the function clean_sm
def clean_sm(x):
    x= np.where(x == 1,1,0)
    return x

#Create a new dataframe for target sm_li
ss = pd.DataFrame({
    'sm_li': clean_sm(s['web1h']),
    'income': np.where((s['income'] < 1) | (s['income'] > 9), np.nan, s['income']),
    'education': np.where((s['educ2'] < 1) | (s['educ2'] > 8), np.nan, s['educ2']),
    'parent': np.where(s['par'] == 1, 1, 0),
    'married': np.where(s['marital'] == 1, 1, 0),
    'female': np.where(s['gender'] == 2, 1, 0),
    'age': np.where(s['age'] > 98, np.nan, s['age'])
})
ss = ss.dropna()

# target vector y
y = ss['sm_li']
#features set exclude y
X = ss[['income','education','parent','married','female','age']]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=987)

# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced', random_state=987)

# Fit algorithm to training data
logreg_model = lr.fit(X_train, y_train)


#Function to predict data from user inputs
def pred_user (lr, data):
    predicted_class = lr.predict([data])
    probs = lr.predict_proba([data])
    # Print predicted class and probability
    result = {
        "predicted_class": predicted_class[0],
        "probability": round(probs[0][1], 3)
    }
    return result

#Streamlit page setup
st.markdown("# Welcome to LinkedIn User Prediction App!")
st.markdown("### Lets Predict LinkedIn User!!")
st.markdown("#### Please fill the values for questions below:")


#streamlit values for (income,edu,parent,married,female,age)
#income 
income_levels = {
    1: "1: Less than $10,000",
    2: "2: $10,000 to under $20,000",
    3: "3: $20,000 to under $30,000",
    4: "4: $30,000 to under $40,000",
    5: "5: $40,00 to under $50,000",
    6: "6: $50,000 to under $75,000",
    7: "7: $75,000 to under $100,000",
    8: "8: $100,000 to under $150,000",
    9: "9: $150,000 or more"
}
income_str = st.selectbox("Select Income Level:", list(income_levels.values()))
income = next(key for key, value in income_levels.items() if value == income_str)

#education
edu_levels = {
    1: "1: Less than high school (Grades 1-8 or no formal schooling)",
    2: "2: High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    3: "3: High school graduate (Grade 12 with diploma or GED certificate)",
    4: "4: Some college, no degree (includes some community college)",
    5: "5: Two-year associate degree from a college or university",
    6: "6: Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    7: "7: Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    8: "8: Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
}
edu_str = st.selectbox("Select Education Level:", list(edu_levels.values()))
edu = next(key for key, value in edu_levels.items() if value == edu_str)

#parent
parent_options = ["Yes", "No/Don't Know"]
parent = st.selectbox("Are you a parent of a child under 18 living in your home?:", parent_options)
par_code = 1 if parent == "Yes" else 0

#marital 
married_options = ["Married", "Living with a partner","Divorced","Separated","Widowed","Never been married", "Don't Know", "Refused"]
married = st.selectbox("Select Martial Status:", married_options)
mar_code = 1 if married == "Married" else 0

#female
gender_options = ["Female", "Male", "Other", "Refused"]
gender = st.selectbox("Select Gender:", gender_options)
gender_code = 1 if gender == "Female" else 0

#age
age = st.slider("Select Age:", min_value=18, max_value=99, value=18)

#combine all values to data vector
data= [income, edu, par_code, mar_code, gender_code, age]

# Call pred_user function and make prediction when the user clicks the button
if st.button("PREDICT"):
    result = pred_user(lr, data)

    # Display the prediction results
    if result['predicted_class'] == 1:
        st.write(f" #### **LinkedIn User!!!! <br> Probability that this person is a LinkedIn User is: {round(result['probability'] * 100, 2)}%**", unsafe_allow_html=True)
    else:  
        st.write(f"#### **Non-LinkedIn User! <br> Probability that this person is a LinkedIn User is: {round(result['probability'] * 100, 2)}%**", unsafe_allow_html=True )

