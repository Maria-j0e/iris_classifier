import streamlit as st
import pandas as pd
from os import path
import numpy as np
import pickle

st.title("Flower Species Predictor:")

#petal_width = st.slider('Petal Width', min_value = 0.1, max_value = 3.0)

petal_length = st.number_input('Please choose Petal length' , min_value = 1.0, max_value = 6.9
                               ,placeholder = 'Enter a value between 1.0  &  6.9', value = None)
petal_width = st.number_input('Please choose Petal width', min_value = 0.1, max_value = 2.5,placeholder = 'Enter a value between 0.1  &  2.5', value = None)
sepal_length = st.number_input('Please choose Sepal length', min_value = 4.3, max_value = 7.9,placeholder = 'Enter a value between 4.3  &  7.9', value = None)
sepal_width = st.number_input('Please choose Sepal width', min_value = 2.0, max_value = 4.4,placeholder = 'Enter a value between 2.0  &  4.4', value = None)

#prepare the dataframe for prediction
df_user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                          columns = ['sepal_length', 'sepal_width','petal_length', 'petal_width' ])

model_path = path.join("Model" , "iris_classifier.pkl")
with open(model_path, 'rb') as file :
    iris_predictor = pickle.load(file)

dict_species = {0:'setosa',1:'versicolor',2:'virginica'}

if st.button("predict species"):
    if((petal_length == None) or (petal_width == None) or (sepal_length == None) or (sepal_width == None)) :
        #will be executed for improper values
        st.write("please fill all values")
    else:
        #prediction can be done here
        predicted_species = iris_predictor.predict(df_user_input)
        #we use the value to find corresponding species
        st.write("the species is",dict_species [predicted_species[0]])
