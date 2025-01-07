import pandas as pd
import streamlit as st
import pickle
#import sklearn
#from sklearn.ensemble import RandomForestClassifier

st.title('Titanic passengers survival prediction')

st.write('This is a simple example of a Machine Learning model deployment using Streamlit.')


user_input_pclass = st.selectbox('Pclass', [1, 2, 3])
user_input_sex = st.selectbox('Sex', ['Male', 'Female'])
user_input_age = st.number_input('Age', min_value=0, max_value=100, value=30)

rfclf = pickle.load(open('rfclf.pkl', 'rb'))
def predict():
    encode_dict = {'Male' : 0, 'Female' : 1}
    data = [[user_input_pclass,encode_dict[user_input_sex], user_input_age, 1,0,0,1,0]]

    result = rfclf.predict(data)

    return result

if st.button('Predict'):
    result = predict()
    if result == 1:
        st.write('Survived')
    else:
        st.write('Not survived')

else:
    st.write('Please input values to predict survival')