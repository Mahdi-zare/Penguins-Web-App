import streamlit as st
import joblib
import pandas as pd
import numpy as np


st.write("""
# Penguins Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [Seaborn datasets](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv) in Seaborn library.
""")

st.sidebar.header("User Input features")

upload_file = st.sidebar.file_uploader("Uplode your csv file:",type=["csv","xlsx"])

if upload_file is not None:
    input_data = pd.read_csv(upload_file)
else:
    def user_input_feature():
        island = st.sidebar.selectbox("Island",['Biscoe','Dream','Torgersen'])
        sex = st.sidebar.selectbox("Sex",["male","female"])
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_data = user_input_feature()

mlpipe = joblib.load("GradientBoostingClassifier.joblib")

st.subheader("User Input fearure")

if upload_file is not None:
    st.write(input_data)
else:
    st.write(input_data)

pred = mlpipe.predict(input_data)
pred_proba = mlpipe.predict_proba(input_data)
prob_df = pd.DataFrame(pred_proba,columns=['Adelie','Chinstrap','Gentoo'])


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[pred])

st.subheader('Prediction Probability')
st.write(prob_df)

st.write("""
#### train and deployed by [Mahdi Zare](https://www.linkedin.com/in/mahdizare22/)
""")
