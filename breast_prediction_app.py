# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:59:06 2025

@author: HP
"""

import os
import numpy as np
import streamlit as st
import pickle

model_path = os.path.join(os.path.dirname(__file__),"breast_cancer.sav")
try:
    breast_cancer = pickle.load(open(model_path,"rb"))
except FileNotFoundError:
    st.error("Model File Not Found")
    st.stop()
    
def Breast_cancer(input_data):
    input_data_as_numpy_array(input_data).reshape(1,-1)
    prediction = breast_cancer.predict(input_data_as_numpy_array)
    if prediction[0] == 0:
        return "The tumor is Malignant (Cancerous)"
    else:
        return "The tumor is Benign (Non-Cancerous)"

def main():
    st.title("Breast Cancer Prediction System")
    st.write("Enter the 30 cells nucleus features below to predict whether the the tumor is Malignant or Benign")
    mean_radius = st.number_input("Mean Radius = ")
    mean_texture = st.number_input("Mean Texture")
    mean_perimeter = st.number_input("Mean Perimeter")
    mean_area = st.number_input("Mean_Area")
    mean_smoothness = st.number_input("Mean Smoothness")
    mean_compactness = st.number_input("Mean Compactness")
    mean_concavity = st.number_input("Mean_Concavity")
    mean_concave_points = st.number_input("Mean Concave Points")
    mean_symmetry = st.number_input("Mean Symmetry")
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension")
    radius_error = st.number_input("Radius Error")
    texture_error = st.number_input("Texture Error")
    perimeter_error = st.number_input("Perimeter Error")
    area_error = st.number_input("Area Error")
    smoothness_error = st.number_input("Smoothness Error")
    compactness_error = st.number_input("Compactness Error")
    concavity_error = st.number_input("Concavity Error")
    concave_points_error = st.number_input("Cpncave Points Error")
    symmetry_error = st.number_input("Symmetry Error")
    fractal_dimension_error = st.number_input("Fractal Dimension Error")
    worst_radius = st.number_input("Worst Radius")
    worst_texture = st.number_input("Worst Texture")
    worst_perimeter = st.number_input("Worst Perimeter")
    worst_area = st.number_input("Worst Area")
    worst_smoothness = st.number_input("Wprst Smoothness")
    worst_compactness = st.number_input("Worst Compactnesss")
    worst_concavity = st.number_input("Worst Concavity")
    worst_concave_points = st.number_input("Worst Concave Points")
    worst_symmetry = st.number_input("Worst Symmetry")
    worst_fractal_dimension = st.number_input("Wprst Fractal Dimension")
    
    diagnosis = ""
    if st.button("Predict Tumor Type"):
        diagnonis = Breast_cancer([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
            mean_fractal_dimension, radius_error, texture_error, perimeter_error,
            area_error, smoothness_error, compactness_error, concavity_error,
            concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
            worst_compactness, worst_concavity, worst_concave_points, worst_symmetry,
            worst_fractal_dimension])
        
    st.success(diagnosis)
    
if __name__ == "__main__":
    main()
        
    
    
    
    
            
                                