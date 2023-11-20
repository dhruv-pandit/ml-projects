import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import seaborn as sns
streamlit_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300&display=swap');

    html, body, [class*="css"] {
    font-family: 'Open Sans';
    }

    h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', serif;
    }
    </style>
"""
st.markdown(streamlit_style, unsafe_allow_html=True)
# Read data
@st.cache_data
def load_data():
    df_wine = pd.read_csv(r'datasets/winequalityN.csv')
    df_results_nan = pd.read_excel(r'datasets/model_performance.xlsx', sheet_name='nan')
    df_results_med = pd.read_excel(r'datasets/model_performance.xlsx', sheet_name='median')
    df_results_best = pd.read_excel(r'datasets/model_performance.xlsx', sheet_name='best')

    return df_wine, df_results_nan, df_results_med, df_results_best

df_wine, df_results_nan, df_results_med, df_results_best = load_data()
df_wine_dropna = df_wine.dropna()
st.title("*Pour* Decisions? Classifier Performance Analysis for Vinho Verde Wine Quality Prediction")
st.write("""
         * Contact at dhruvpandit@aln.iseg.ulisboa.pt
""")
st.write("""
         ## Dataset Description

Welcome to my Vinho Verde Wine Quality Classifier project! In this exploration of machine learning, I'm working with datasets curated by Paulo Cortez to classify wine quality. Explore how different classifiers perform based on physicochemical attributes like acidity, sulfur dioxide levels, and more.

Source: Paulo Cortez, University of Minho, Guimarães, Portugal, [http://www3.dsi.uminho.pt/pcortez](http://www3.dsi.uminho.pt/pcortez) A. Cerdeira, F. Almeida, T. Matos, and J. Reis, Viticulture Commission of the Vinho Verde Region (CVRVV), Porto, Portugal @2009

Data Set Information: [Cortez et al., 2009]. Input variables: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, alcohol, and quality (score between 0 and 10) as the output variable based on sensory data.

        
""")

st.dataframe(df_wine.describe().round(2))  # Same as st.write(df)


categorical_cols_median = df_wine.select_dtypes(include=['object']).columns
numeric_cols_median = df_wine.select_dtypes(include=[np.number]).columns




fig = go.Figure()

for each in numeric_cols_median:
    fig.add_trace(go.Box(y=df_wine[each], name=each.capitalize()))  # Set 'name' to the column name or any custom label

fig.update_layout(
    xaxis_title="Variable",  # Customize the x-axis title
    yaxis_title="Value",     # Customize the y-axis title
)
st.write("""### Distribution of Wine Characteristics
Each box represents the distribution of key physicochemical variables, such as acidity, sulfur dioxide levels, and more. The box's central line denotes the median, while the box itself spans the interquartile range (IQR). Whiskers extend to the minimum and maximum values within 1.5 times the IQR. Outliers, depicted as individual points, stand out beyond this range. Use these box plots to discern the central tendencies, variability, and potential outliers in each variable. A wider spread or skewed distribution may indicate critical features influencing wine quality. Explore the plots to glean insights into the unique characteristics defining Vinho Verde wines.
         """)
st.plotly_chart(fig, use_container_width = True)

# ave_qu = df_wine.groupby("quality").mean()
# st.plotly_chart(ave_qu.plot(kind="bar",figsize=(20,10))

option = st.selectbox('Choose a method for dealing with NaN values', options=['Median', 'Nan'])
if option == 'Nan':
    fig_results = px.bar(df_results_med, x='Model', y='RMSE', color = 'RMSE',color_continuous_scale= 'oranges')
    title='Dropping NaN Values'
else:
    fig_results = px.bar(df_results_nan, x='Model', y='RMSE', color = 'RMSE',color_continuous_scale= 'oranges')
    title='Filling NaN With Median Values'  
fig_results.data[0].hovertemplate = f'Model: %{{x}} <br>RMSE: %{{y:.3f}}' 

st.write("""
         ### Comparing Model Performances with Null Handling:
In this dropdown menu, choose between two strategies: dropping Null values or filling them with median values. As you navigate through the different machine learning models, observe how RMSE scores vary based on the selected approach. Keep an eye on the trends — you might notice that filling Null values with medians tends to yield lower RMSE scores. This comparison sheds light on the influence of data preprocessing choices on the accuracy of our predictive models. 
         """)
st.write(f"#### {title}")    
st.plotly_chart(fig_results, use_container_width = True)

fig_best = px.bar(df_results_best, x='Model', y='RMSE', color = 'RMSE',color_continuous_scale= 'PuOR')
fig_best.data[0].hovertemplate = f'Model: %{{x}} <br>RMSE: %{{y:.3f}}' 

st.write("""
### Model Fine-Tuning: Unlocking the Best Configurations

In our quest for better models, we've fine-tuned the hyperparameters of four powerful models: Decision Tree, ADABoost, Random Forest, and XGBoost. Notice not just the reduction in RMSE for each model, but the improvements made to the XGBoost Model performance in particular. 
##### Decision Tree Tuning

Our Decision Tree model undergoes meticulous tuning to optimize its performance. The hyperparameter exploration includes variations in maximum depth, minimum samples for split and leaf, and features. Employing a grid search approach, we systematically analyze different configurations to uncover the set of parameters that yields the most accurate predictions for Vinho Verde wine quality.

##### ADABoost Tuning

ADABoost, a robust ensemble method, is fine-tuned to enhance its accuracy in predicting wine quality. Through a grid search, we investigate the optimal combination of the number of estimators, learning rate, and the base estimator, which is a Decision Tree with varying depths. This process ensures that ADABoost adapts seamlessly to the intricacies of the Vinho Verde dataset, providing improved predictive capabilities.

##### Random Forest Tuning 

Our Random Forest model undergoes precision crafting through hyperparameter tuning. Key parameters such as the number of estimators, maximum depth, minimum samples for split and leaf are scrutinized. Utilizing a grid search strategy, we systematically explore diverse configurations to identify the ideal settings. The outcome is a finely-tuned Random Forest model equipped to discern subtle patterns in the Vinho Verde wine dataset.

##### XGBoost Tuning

XGBoost, a powerful gradient boosting algorithm, is refined to unleash its maximum predictive power. The tuning process encompasses parameters such as the number of estimators, learning rate, maximum depth, and subsampling ratio. Through a meticulous grid search, we navigate the parameter space to pinpoint the combination that optimizes XGBoost's ability to capture the nuances of Vinho Verde wine quality    
         """)
st.write("#### Model Comparison After Tuning Hyperparameters")
st.plotly_chart(fig_best, use_container_width = True)

