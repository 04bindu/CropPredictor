import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
   
    st.subheader(" Find out the most suitable crop to grow in your farm  ")
    N = st.number_input("Nitrogen", 1, 10000)
    P = st.number_input("Phosporus", 1, 10000)
    K = st.number_input("Potassium", 1, 10000)
    temp = st.number_input("Temperature", 0.0, 100000.0)
    humidity = st.number_input("Humidity in %", 0.0, 100000.0)
    ph = st.number_input("Ph", 0.0,100000.0)
    rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1,-1)
        
    if st.button('Predict'):

        loaded_model = load_model('model.pkl')
        prediction = loaded_model.predict(single_pred)
        st.write('''
		    ## Results 
		    ''')
        st.success(f"{prediction.item().title()} is recommended for your field")
      

if __name__ == '__main__':
	main()