import streamlit as st
import numpy as np
import pickle  


with open('model.pkl', 'rb') as file:
    dt = pickle.load(file)

# Define the predict function
def predict(features):
    return dt.predict(np.array(features).reshape(1, -1))

# Define feature names (replace with the actual feature names in your dataset)
feature_names = [

    "GENDER", "ENROLLED_UNIVERSITY", "EDUCATION_LEVEL", 
    "MAJOR_DISCIPLINE", "COMPANY_SIZE", "COMPANY_TYPE", "COMPANY_SIZE"
]

# Streamlit app title
st.title('Employee Retention Prediction')

# Create input fields for each feature
user_input = []
for feature in feature_names:
    value = st.number_input(f'{feature}', value=0.0)
    user_input.append(value)

# Create a button for making predictions
if st.button('Predict'):
    prediction = predict(user_input)
    st.success(f'Predicted Segment: {prediction[0]}')
