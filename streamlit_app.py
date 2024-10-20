# import streamlit as st
# import pandas as pd

# st.title('🤖 Machine Learning App')

# st.info('This app explores child undernutrition data and builds a predictive machine learning model.')

# # Load the data
# df = pd.read_csv('https://raw.githubusercontent.com/getnetbogale27/Child-Undernutrition-ML-app/master/data/ML-Data.csv?token=GHSAT0AAAAAACWZVWX77CD73OUBBABGGOB2ZW6WBFA')

# # Expanders for different data views
# with st.expander('Raw data (first 5 rows)'):
#     st.write(df.head(5))  # Display first 5 rows of raw data

# with st.expander('X (independent variables) (first 5 rows)'):
#     X_raw = df.iloc[:, 3:-1]
#     st.write(X_raw.head(5))  # Display first 5 rows of independent variables

# with st.expander('y (dependent variable) (first 5 rows)'):
#     y_raw = df.iloc[:, -1]
#     st.write(y_raw.head(5))  # Display first 5 rows of dependent variable

# # with st.expander('Data visualization'):
# #     st.write('Scatter plot: Child Age vs Caregiver Age')
# #     st.write('Note: Unable to color by Nutrition Status directly. You can filter or group data for more specific visualizations.')
# #     st.scatter_chart(df[['Ch_age_mon', 'Care_age']])

# with st.expander('Data visualization'):
#   st.scatter_chart(data=df, x='Ch_age_mon', y='Care_age', color='Nutrition_Status')

# # Input features
# with st.sidebar:
#   st.header('Input features')
#   Region = st.selectbox('Region', ('Tigray', 'Amhara', 'Oromia', 'SNNP', 'Addis Ababa'))
#   Residence = st.selectbox('Residence', ('Urban', 'Rural'))
#   Ch_Sex = st.selectbox('Child Gender', ('Male', 'Female'))
#   Ch_age_mon = st.slider('Child Age (Month)', 4, 192, 45)
#   Ch_Age = st.selectbox('Child Age', ('Infancy', 'Early childhood', 'Middle childhood/ Preadolescence', 'Adolescence'))  
#   Ch_longterm_health_problem = st.selectbox('Child has longterm health problem', ('No', 'Yes'))

  

############################
############################
# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# # from sklearn.preprocessing import LabelEncoder


# st.title('🤖 Machine Learning App')

# st.info('This app explores child undernutrition data and builds a predictive machine learning model.')

# # Load the data
# df = pd.read_csv('https://raw.githubusercontent.com/getnetbogale27/Child-Undernutrition-ML-app/master/data/ML-Data.csv?token=GHSAT0AAAAAACWZVWX7LM7EA4CGQXGX52NQZW6ZYWA')

# # Expanders for different data views
# with st.expander('Raw data (first 5 rows)'):
#     st.write(df.head(5))  # Display first 5 rows of raw data

# with st.expander('X (independent variables) (first 5 rows)'):
#     X_raw = df.iloc[:, 3:-1]
#     st.write(X_raw.head(5))  # Display first 5 rows of independent variables

# with st.expander('y (dependent variable) (first 5 rows)'):
#     y_raw = df.iloc[:, -1]
#     st.write(y_raw.head(5))  # Display first 5 rows of dependent variable

# # with st.expander('Data visualization'):
# #     st.write('Scatter plot: Child Age vs Caregiver Age')
# #     st.write('Note: Unable to color by Nutrition Status directly. You can filter or group data for more specific visualizations.')
# #     st.scatter_chart(df[['Ch_age_mon', 'Care_age']])

# with st.expander('Data visualization'):
#   st.scatter_chart(data=df, x='Ch_age_mon', y='Care_age', color='Nutrition_Status')

# # # Input features
# # with st.sidebar:
# #   st.header('Input features')
# #   Region = st.selectbox('Region', ('Tigray', 'Amhara', 'Oromia', 'SNNP', 'Addis Ababa'))
# #   Residence = st.selectbox('Residence', ('Urban', 'Rural'))
# #   Ch_Sex = st.selectbox('Child Gender', ('Male', 'Female'))
# #   Ch_age_mon = st.slider('Child Age (Month)', 4, 192, 45)
# #   Ch_Age = st.selectbox('Child Age', ('Infancy', 'Early childhood', 'Middle childhood/ Preadolescence', 'Adolescence'))
# #   Ch_longterm_health_problem = st.selectbox('Child has longterm health problem', ('No', 'Yes'))


# # Sidebar inputs
# with st.sidebar:
#     st.header('Input features')

# #     # Input features Continous and categorical variables
# #     region = st.selectbox('Region', ['Tigray', 'Amhara', 'Oromiya', 'SNNP', 'Addis Ababa City Administration'])
# #     residence = st.selectbox('Residence', ['Urban', 'Rural'])
# #     ch_sex = st.selectbox('Child Sex', ['Male', 'Female'])
# #     ch_age_mon = st.slider('Child Age in Months', 4, 192, 50, step=1)
# #     ch_age_category = st.selectbox('Child Age Category', ['Infancy', 'Early childhood', 'Middle childhood/ Preadolescence', 'Adolescence'])
# #     ch_longterm_health_problem = st.selectbox('Long-term Health Problem', ['No', 'Yes'])
# #     ch_health_compared_peers = st.selectbox('Health Compared to Peers', ['Same', 'Better', 'Worse'])
# #     ch_health_general_new = st.selectbox('General Health', ['Poor', 'Average', 'Good'])
# #     ch_injury = st.selectbox('Child Injury', ['No', 'Yes'])
# #     csw = st.selectbox('Child\'s Social Welfare (CSW)', ['Low', 'High'])
# #     care_edu_new = st.selectbox('Caregiver Education', ['Illiterate', 'Literate'])
# #     care_rln_head = st.selectbox('Caregiver Relationship to Household Head', ['Caregiver is household head', 'Caregiver is partner of household head'])
# #     care_age = st.slider('Caregiver Age', 8, 87, 30, step=1)
# #     care_sex = st.selectbox('Caregiver Sex', ['Male', 'Female'])
# #     care_relat = st.selectbox('Caregiver Relation', ['Biological parent', 'Non-biological parent'])
# #     cgws = st.selectbox('Caregiver Social Welfare (CSW)', ['Low', 'High'])
# #     dad_age = st.slider('Dad Age', 19, 90, 40, step=1)
# #     dad_edu_new = st.selectbox('Dad Education', ['Illiterate', 'Literate'])
# #     dad_live_location = st.selectbox('Dad Live Location', ['Lives in the household', 'Does not live in household', 'Has died'])
# #     mom_live_location = st.selectbox('Mom Live Location', ['Lives in the household', 'Does not live in household'])
# #     mom_age = st.slider('Mom Age', 15, 64, 35, step=1)
# #     mom_edu_new = st.selectbox('Mom Education', ['Illiterate', 'Literate'])
# #     head_edu_new = st.selectbox('Household Head Education', ['Illiterate', 'Literate'])
# #     head_age = st.slider('Household Head Age', 5, 110, 45, step=1)
# #     head_sex = st.selectbox('Household Head Sex', ['Male', 'Female'])
# #     head_relation = st.selectbox('Household Head Relation', ['Biological Parent', 'Non-biological Parent'])
# #     household_size = st.selectbox('Household Size', ['<=6', '>6'])
# #     wealth_quintile = st.selectbox('Wealth Quintile', ['Poorest', 'Secondary', 'Middle', 'Fourth', 'Wealthiest'])
# #     access_to_safe_drinking_water = st.selectbox('Access to Safe Drinking Water', ['No', 'Yes'])
# #     access_to_toilet = st.selectbox('Access to Toilet', ['No', 'Yes'])
# #     access_to_electricity = st.selectbox('Access to Electricity', ['No', 'Yes'])
# #     acces_to_cooking_fuels = st.selectbox('Access to Cooking Fuels', ['No', 'Yes'])

# # # Define a mapping function for categorical values to numerical codes
# # def get_numeric_code(category, category_type):
# #     mappings = {
# #         'Region': {'Tigray': 1, 'Amhara': 3, 'Oromiya': 4, 'SNNP': 7, 'Addis Ababa City Administration': 14},
# #         'Residence': {'Urban': 1, 'Rural': 2},
# #         'Ch_sex': {'Male': 1, 'Female': 2},
# #         'Ch_Age_Category': {'Infancy': 1, 'Early childhood': 2, 'Middle childhood/ Preadolescence': 3, 'Adolescence': 4},
# #         'Ch_longterm_health_problem': {'No': 1, 'Yes': 2},
# #         'Ch_health_compared_peers': {'Same': 1, 'Better': 2, 'Worse': 3},
# #         'Ch_health_general_new': {'Poor': 1, 'Average': 2, 'Good': 3},
# #         'Ch_injury': {'No': 1, 'Yes': 2},
# #         'CSW': {'Low': 1, 'High': 2},
# #         'Care_Edu_New': {'Illiterate': 1, 'Literate': 2},
# #         'Care_rln_head': {'Caregiver is household head': 1, 'Caregiver is partner of household head': 2},
# #         'Care_sex': {'Male': 1, 'Female': 2},
# #         'Care_relat': {'Biological parent': 1, 'Non-biological parent': 2},
# #         'CGSW': {'Low': 1, 'High': 2},
# #         'Dad_Edu_New': {'Illiterate': 1, 'Literate': 2},
# #         'Dad_live_location': {'Lives in the household': 1, 'Does not live in household': 2, 'Has died': 3},
# #         'Mom_live_location': {'Lives in the household': 1, 'Does not live in household': 2},
# #         'Mom_Edu_New': {'Illiterate': 1, 'Literate': 2},
# #         'Head_Edu_New': {'Illiterate': 1, 'Literate': 2},
# #         'Head_sex': {'Male': 1, 'Female': 2},
# #         'Head_Relation': {'Biological Parent': 1, 'Non-biological Parent': 2},
# #         'Household_size': {'<=6': 1, '>6': 2},
# #         'Wealth_quintile': {'Poorest': 1, 'Secondary': 2, 'Middle': 3, 'Fourth': 4, 'Wealthiest': 5},
# #         'Access_to_safe_drinking_water': {'No': 1, 'Yes': 2},
# #         'Access_to_toilet': {'No': 1, 'Yes': 2},
# #         'Access_to_electricity': {'No': 1, 'Yes': 2},
# #         'Acces_to_cooking_fuels': {'No': 1, 'Yes': 2}
# #     }
# #     return mappings[category_type][category]

# # # Use the function to get the numeric codes
# # region_code = get_numeric_code(region, 'Region')
# # residence_code = get_numeric_code(residence, 'Residence')
# # ch_sex_code = get_numeric_code(ch_sex, 'Ch_sex')
# # ch_age_category_code = get_numeric_code(ch_age_category, 'Ch_Age_Category')
# # ch_longterm_health_problem_code = get_numeric_code(ch_longterm_health_problem, 'Ch_longterm_health_problem')
# # ch_health_compared_peers_code = get_numeric_code(ch_health_compared_peers, 'Ch_health_compared_peers')
# # ch_health_general_new_code = get_numeric_code(ch_health_general_new, 'Ch_health_general_new')
# # ch_injury_code = get_numeric_code(ch_injury, 'Ch_injury')
# # csw_code = get_numeric_code(csw, 'CSW')
# # care_edu_new_code = get_numeric_code(care_edu_new, 'Care_Edu_New')
# # care_rln_head_code = get_numeric_code(care_rln_head, 'Care_rln_head')
# # care_sex_code = get_numeric_code(care_sex, 'Care_sex')
# # care_relat_code = get_numeric_code(care_relat, 'Care_relat')
# # cgws_code = get_numeric_code(cgws, 'CGSW')
# # dad_age_code = dad_age
# # dad_edu_new_code = get_numeric_code(dad_edu_new, 'Dad_Edu_New')
# # dad_live_location_code = get_numeric_code(dad_live_location, 'Dad_live_location')
# # mom_live_location_code = get_numeric_code(mom_live_location, 'Mom_live_location')
# # mom_age_code = mom_age
# # mom_edu_new_code = get_numeric_code(mom_edu_new, 'Mom_Edu_New')
# # head_edu_new_code = get_numeric_code(head_edu_new, 'Head_Edu_New')
# # head_age_code = head_age
# # head_sex_code = get_numeric_code(head_sex, 'Head_sex')
# # head_relation_code = get_numeric_code(head_relation, 'Head_Relation')
# # household_size_code = get_numeric_code(household_size, 'Household_size')
# # wealth_quintile_code = get_numeric_code(wealth_quintile, 'Wealth_quintile')
# # access_to_safe_drinking_water_code = get_numeric_code(access_to_safe_drinking_water, 'Access_to_safe_drinking_water')
# # access_to_toilet_code = get_numeric_code(access_to_toilet, 'Access_to_toilet')
# # access_to_electricity_code = get_numeric_code(access_to_electricity, 'Access_to_electricity')
# # acces_to_cooking_fuels_code = get_numeric_code(acces_to_cooking_fuels, 'Acces_to_cooking_fuels')

# # # Display or use the inputs
# # input_data = {
# #     'region': region_code,
# #     'residence': residence_code,
# #     'ch_sex': ch_sex_code,
# #     'ch_age_mon': ch_age_mon,
# #     'ch_age_category': ch_age_category_code,
# #     'ch_longterm_health_problem': ch_longterm_health_problem_code,
# #     'ch_health_compared_peers': ch_health_compared_peers_code,
# #     'ch_health_general_new': ch_health_general_new_code,
# #     'ch_injury': ch_injury_code,
# #     'csw': csw_code,
# #     'care_edu_new': care_edu_new_code,
# #     'care_rln_head': care_rln_head_code,
# #     'care_sex': care_sex_code,
# #     'care_relat': care_relat_code,
# #     'cgws': cgws_code,
# #     'dad_age': dad_age_code,
# #     'dad_edu_new': dad_edu_new_code,
# #     'dad_live_location': dad_live_location_code,
# #     'mom_live_location': mom_live_location_code,
# #     'mom_age': mom_age_code,
# #     'mom_edu_new': mom_edu_new_code,
# #     'head_edu_new': head_edu_new_code,
# #     'head_age': head_age_code,
# #     'head_sex': head_sex_code,
# #     'head_relation': head_relation_code,
# #     'household_size': household_size_code,
# #     'wealth_quintile': wealth_quintile_code,
# #     'access_to_safe_drinking_water': access_to_safe_drinking_water_code,
# #     'access_to_toilet': access_to_toilet_code,
# #     'access_to_electricity': access_to_electricity_code,
# #     'acces_to_cooking_fuels': acces_to_cooking_fuels_code
# # }

# # # Print or use the input data
# # st.write("Input data for model:", input_data)

# # Sidebar inputs
# with st.sidebar:
#     st.header('Input features')
#     # Input features Continous and categorical variables
#     region = st.selectbox('Region', ['Tigray', 'Amhara', 'Oromiya', 'SNNP', 'Addis Ababa City Administration'], index=0)
#     residence = st.selectbox('Residence', ['Urban', 'Rural'], index=0)
#     ch_sex = st.selectbox('Child Sex', ['Male', 'Female'], index=0)
#     ch_age_mon = st.slider('Child Age in Months', 4, 192, 50, step=1)
#     ch_age_category = st.selectbox('Child Age Category', ['Infancy', 'Early childhood', 'Middle childhood/ Preadolescence', 'Adolescence'], index=0)
#     ch_longterm_health_problem = st.selectbox('Long-term Health Problem', ['No', 'Yes'], index=0)
#     ch_health_compared_peers = st.selectbox('Health Compared to Peers', ['Same', 'Better', 'Worse'], index=0)
#     ch_health_general_new = st.selectbox('General Health', ['Poor', 'Average', 'Good'], index=0)
#     ch_injury = st.selectbox('Child Injury', ['No', 'Yes'], index=0)
#     csw = st.selectbox('Child\'s Social Welfare (CSW)', ['Low', 'High'], index=0)
#     care_edu_new = st.selectbox('Caregiver Education', ['Illiterate', 'Literate'], index=0)
#     care_rln_head = st.selectbox('Caregiver Relationship to Household Head', ['Caregiver is household head', 'Caregiver is partner of household head'], index=0)
#     care_age = st.slider('Caregiver Age', 8, 87, 30, step=1)
#     care_sex = st.selectbox('Caregiver Sex', ['Male', 'Female'], index=0)
#     care_relat = st.selectbox('Caregiver Relation', ['Biological parent', 'Non-biological parent'], index=0)
#     cgws = st.selectbox('Caregiver Social Welfare (CSW)', ['Low', 'High'], index=0)
#     dad_age = st.slider('Dad Age', 19, 90, 40, step=1)
#     dad_edu_new = st.selectbox('Dad Education', ['Illiterate', 'Literate'], index=0)
#     dad_live_location = st.selectbox('Dad Live Location', ['Lives in the household', 'Does not live in household', 'Has died'], index=0)
#     mom_live_location = st.selectbox('Mom Live Location', ['Lives in the household', 'Does not live in household'], index=0)
#     mom_age = st.slider('Mom Age', 15, 64, 35, step=1)
#     mom_edu_new = st.selectbox('Mom Education', ['Illiterate', 'Literate'], index=0)
#     head_edu_new = st.selectbox('Household Head Education', ['Illiterate', 'Literate'], index=0)
#     head_age = st.slider('Household Head Age', 5, 110, 45, step=1)
#     head_sex = st.selectbox('Household Head Sex', ['Male', 'Female'], index=0)
#     head_relation = st.selectbox('Household Head Relation', ['Biological Parent', 'Non-biological Parent'], index=0)
#     household_size = st.selectbox('Household Size', ['<=6', '>6'], index=0)
#     wealth_quintile = st.selectbox('Wealth Quintile', ['Poorest', 'Secondary', 'Middle', 'Fourth', 'Wealthiest'], index=0)
#     access_to_safe_drinking_water = st.selectbox('Access to Safe Drinking Water', ['No', 'Yes'], index=0)
#     access_to_toilet = st.selectbox('Access to Toilet', ['No', 'Yes'], index=0)
#     access_to_electricity = st.selectbox('Access to Electricity', ['No', 'Yes'], index=0)
#     acces_to_cooking_fuels = st.selectbox('Access to Cooking Fuels', ['No', 'Yes'], index=0)


# # Define X and y
# X = df.iloc[:, 3:-1]  # Features (independent variables)
# y = df['Nutrition_Status']  # Target (dependent variable)

# # Train the ML model
# clf = RandomForestClassifier()
# clf.fit(X, y)

# # Prepare input data for prediction
# input_row = input_df.copy()

# # Apply model to make predictions
# prediction = clf.predict(input_row)
# prediction_proba = clf.predict_proba(input_row)

# # Prepare DataFrame for prediction probabilities
# df_prediction_proba = pd.DataFrame(prediction_proba, columns=['N', 'U', 'S', 'W', 'US', 'UW', 'SW', 'USW'])

# # Rename columns if needed (Example renaming based on prediction_proba indices)
# df_prediction_proba.rename(columns={0: 'N',
#                                     1: 'U',
#                                     2: 'S',
#                                     3: 'W',
#                                     4: 'US',
#                                     5: 'UW',
#                                     6: 'SW',
#                                     7: 'USW'}, inplace=True)

# # Display predictions
# with st.expander('Model Inference'):
#     st.write('**Prediction (Class Label)**')
#     st.write(prediction[0])

#     st.write('**Prediction Probabilities**')
#     st.write(df_prediction_proba)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import scipy.stats as stats


# Cache the data loading function


@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/getnetbogale27/Child-Undernutrition-ML-app/refs/heads/master/data/ML-Data.csv?token=GHSAT0AAAAAACY5XTINZM2YD7A7H47DSQ3IZYTUMRQ'
    df = pd.read_csv(url)
    return df


# Load the data
df = load_data()

# Streamlit app title
st.title('🤖 Machine Learning App')

st.info('This app explores child undernutrition data and builds a predictive machine learning model.')

# Expanders for different data views
with st.expander('🔢 Raw data (first 5 rows)'):
    st.write(df.head(5))  # Display first 5 rows of raw data

with st.expander('🧩 X (independent variables) (first 5 rows)'):
    X_raw = df.iloc[:, 3:-1]
    st.write(X_raw.head(5))  # Display first 5 rows of independent variables

with st.expander('🎯 Y (dependent variable) (first 5 rows)'):
    y_raw = df.iloc[:, -1]
    # Display first 5 rows of dependent variable
    st.write(y_raw.head(5).reset_index(drop=True))


# Sidebar inputs
with st.sidebar:
    st.header('Input features')

    # Input features Continous and categorical variables
    region = st.selectbox('Region', [
                          'Tigray', 'Amhara', 'Oromiya', 'SNNP', 'Addis Ababa City Administration'], index=0)
    residence = st.selectbox('Residence', ['Urban', 'Rural'], index=0)
    ch_sex = st.selectbox('Child Sex', ['Male', 'Female'], index=0)
    ch_age_mon = st.slider('Child Age in Months', 4, 192, 50, step=1)
    ch_age_category = st.selectbox('Child Age Category', [
                                   'Infancy', 'Early childhood', 'Middle childhood/ Preadolescence', 'Adolescence'], index=0)
    ch_longterm_health_problem = st.selectbox(
        'Long-term Health Problem', ['No', 'Yes'], index=0)
    ch_health_compared_peers = st.selectbox('Health Compared to Peers', [
                                            'Same', 'Better', 'Worse'], index=0)
    ch_health_general_new = st.selectbox(
        'General Health', ['Poor', 'Average', 'Good'], index=0)
    ch_injury = st.selectbox('Child Injury', ['No', 'Yes'], index=0)
    csw = st.selectbox('Child\'s Social Welfare (CSW)',
                       ['Low', 'High'], index=0)
    care_edu_new = st.selectbox('Caregiver Education', [
                                'Illiterate', 'Literate'], index=0)
    care_rln_head = st.selectbox('Caregiver Relationship to Household Head', [
                                 'Caregiver is household head', 'Caregiver is partner of household head'], index=0)
    care_age = st.slider('Caregiver Age', 8, 87, 30, step=1)
    care_sex = st.selectbox('Caregiver Sex', ['Male', 'Female'], index=0)
    care_relat = st.selectbox('Caregiver Relation', [
                              'Biological parent', 'Non-biological parent'], index=0)
    cgws = st.selectbox('Caregiver Social Welfare (CGWS)',
                        ['Low', 'High'], index=0)
    dad_age = st.slider('Dad Age', 19, 90, 40, step=1)
    dad_edu_new = st.selectbox(
        'Dad Education', ['Illiterate', 'Literate'], index=0)
    dad_live_location = st.selectbox('Dad Live Location', [
                                     'Lives in the household', 'Does not live in household', 'Has died'], index=0)
    mom_live_location = st.selectbox('Mom Live Location', [
                                     'Lives in the household', 'Does not live in household'], index=0)
    mom_age = st.slider('Mom Age', 15, 64, 35, step=1)
    mom_edu_new = st.selectbox(
        'Mom Education', ['Illiterate', 'Literate'], index=0)
    head_edu_new = st.selectbox('Household Head Education', [
                                'Illiterate', 'Literate'], index=0)
    head_age = st.slider('Household Head Age', 5, 110, 45, step=1)
    head_sex = st.selectbox('Household Head Sex', ['Male', 'Female'], index=0)
    head_relation = st.selectbox('Household Head Relation', [
                                 'Biological Parent', 'Non-biological Parent'], index=0)
    household_size = st.selectbox('Household Size', ['<=6', '>6'], index=0)
    wealth_quintile = st.selectbox('Wealth Quintile', [
                                   'Poorest', 'Secondary', 'Middle', 'Fourth', 'Wealthiest'], index=0)
    access_to_safe_drinking_water = st.selectbox(
        'Access to Safe Drinking Water', ['No', 'Yes'], index=0)
    access_to_toilet = st.selectbox('Access to Toilet', ['No', 'Yes'], index=0)
    access_to_electricity = st.selectbox(
        'Access to Electricity', ['No', 'Yes'], index=0)
    access_to_cooking_fuels = st.selectbox(
        'Access to Cooking Fuels', ['No', 'Yes'], index=0)

# Convert inputs to numeric codes


def encode_inputs(X_columns):
    region_map = {'Tigray': 1, 'Amhara': 3, 'Oromiya': 4,
                  'SNNP': 7, 'Addis Ababa City Administration': 14}
    residence_map = {'Urban': 1, 'Rural': 2}
    ch_sex_map = {'Male': 1, 'Female': 2}
    ch_age_category_map = {'Infancy': 1, 'Early childhood': 2,
                           'Middle childhood/ Preadolescence': 3, 'Adolescence': 4}
    ch_longterm_health_problem_map = {'No': 1, 'Yes': 2}
    ch_health_compared_peers_map = {'Same': 1, 'Better': 2, 'Worse': 3}
    ch_health_general_new_map = {'Poor': 1, 'Average': 2, 'Good': 3}
    ch_injury_map = {'No': 1, 'Yes': 2}
    csw_map = {'Low': 1, 'High': 2}
    care_edu_new_map = {'Illiterate': 1, 'Literate': 2}
    care_rln_head_map = {'Caregiver is household head': 1,
                         'Caregiver is partner of household head': 2}
    care_sex_map = {'Male': 1, 'Female': 2}
    care_relat_map = {'Biological parent': 1, 'Non-biological parent': 2}
    cgws_map = {'Low': 1, 'High': 2}
    dad_edu_new_map = {'Illiterate': 1, 'Literate': 2}
    dad_live_location_map = {'Lives in the household': 1,
                             'Does not live in household': 2, 'Has died': 3}
    mom_live_location_map = {
        'Lives in the household': 1, 'Does not live in household': 2}
    mom_edu_new_map = {'Illiterate': 1, 'Literate': 2}
    head_edu_new_map = {'Illiterate': 1, 'Literate': 2}
    head_sex_map = {'Male': 1, 'Female': 2}
    head_relation_map = {'Biological Parent': 1, 'Non-biological Parent': 2}
    household_size_map = {'<=6': 1, '>6': 2}
    wealth_quintile_map = {'Poorest': 1, 'Secondary': 2,
                           'Middle': 3, 'Fourth': 4, 'Wealthiest': 5}
    access_to_safe_drinking_water_map = {'No': 1, 'Yes': 2}
    access_to_toilet_map = {'No': 1, 'Yes': 2}
    access_to_electricity_map = {'No': 1, 'Yes': 2}
    access_to_cooking_fuels_map = {'No': 1, 'Yes': 2}

    input_data = pd.DataFrame({
        'Region': [region_map[region]],
        'Residence': [residence_map[residence]],
        'Ch_sex': [ch_sex_map[ch_sex]],
        'Ch_age_mon': [ch_age_mon],
        'Ch_Age_Category': [ch_age_category_map[ch_age_category]],
        'Ch_longterm_health_problem': [ch_longterm_health_problem_map[ch_longterm_health_problem]],
        'Ch_health_compared_peers': [ch_health_compared_peers_map[ch_health_compared_peers]],
        'Ch_health_general_new': [ch_health_general_new_map[ch_health_general_new]],
        'Ch_injury': [ch_injury_map[ch_injury]],
        'CSW': [csw_map[csw]],
        'Care_Edu_New': [care_edu_new_map[care_edu_new]],
        'Care_rln_head': [care_rln_head_map[care_rln_head]],
        'Care_age': [care_age],
        'Care_sex': [care_sex_map[care_sex]],
        'Care_relat': [care_relat_map[care_relat]],
        'CGSW': [cgws_map[cgws]],
        'Dad_age': [dad_age],
        'Dad_Edu_New': [dad_edu_new_map[dad_edu_new]],
        'Dad_live_location': [dad_live_location_map[dad_live_location]],
        'Mom_live_location': [mom_live_location_map[mom_live_location]],
        'Mom_age': [mom_age],
        'Mom_Edu_New': [mom_edu_new_map[mom_edu_new]],
        'Head_Edu_New': [head_edu_new_map[head_edu_new]],
        'Head_age': [head_age],
        'Head_sex': [head_sex_map[head_sex]],
        'Head_Relation': [head_relation_map[head_relation]],
        'Household_size': [household_size_map[household_size]],
        'Wealth_quintile': [wealth_quintile_map[wealth_quintile]],
        'Access_to_safe_drinking_water': [access_to_safe_drinking_water_map[access_to_safe_drinking_water]],
        'Access_to_toilet': [access_to_toilet_map[access_to_toilet]],
        'Access_to_electricity': [access_to_electricity_map[access_to_electricity]],
        'Access_to_cooking_fuels': [access_to_cooking_fuels_map[access_to_cooking_fuels]],
    })

    # Ensure the DataFrame has the same columns as the training data
    input_data = input_data.reindex(columns=X_columns, fill_value=0)
    return input_data


with st.expander('📈 Data visualization'):
    st.scatter_chart(data=df, x='Ch_age_mon', y='Care_age',
                     color='Nutrition_Status')


# Split data for training and testing
X = df.iloc[:, 3:-1]  # Features (independent variables)
y = df['Nutrition_Status']  # Target (dependent variable)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Cache the model training


@st.cache_data
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


model = train_model(X_train, y_train)

# Prepare input data
X_columns = X.columns
input_data = encode_inputs(X_columns)

# Predict and display the result
if st.button('👉 Click Me to Predict Nutrition Status'):
    # Predict probabilities for each class
    prediction_proba = model.predict_proba(input_data)[0]

    # Predicted Probabilities Expander
    with st.expander('📊 Predicted Probabilities'):
        st.subheader('Predicted Probabilities')

        # Define the category labels
        categories = ['Normal (N)', 'Underweight Only (U)', 'Stunted Only (S)', 'Wasted Only (W)',
                      'Underweight and Stunted (US)', 'Underweight and Wasted (UW)',
                      'Stunted and Wasted (SW)', 'Underweight, Stunted and Wasted (USW)']

        # Create a DataFrame for displaying probabilities
        df_prediction_proba = pd.DataFrame({
            'Category': categories,
            'Probability': prediction_proba
        })

        # Display probabilities using Streamlit's dataframe component
        st.dataframe(df_prediction_proba,
                     column_config={
                         'Category': st.column_config.TextColumn('Category', width='medium'),
                         'Probability': st.column_config.ProgressColumn(
                             'Probability',
                             format='%f',
                             width='medium',
                             min_value=0,
                             max_value=1
                         )
                     }, hide_index=True)

        # Display the predicted category with the highest probability
        predicted_category = categories[np.argmax(prediction_proba)]
        st.success(f'Predicted Category: {predicted_category}')

# Model Evaluation Expander
with st.expander('🧠 Model Evaluation'):
    st.subheader('Model Evaluation')

    # Predict and evaluate the model
    y_pred = model.predict(X_test)

    # Display accuracy score
    st.write(f'Accuracy Score: {accuracy_score(y_test, y_pred):.2f}')

    # Display classification report
    st.write('Classification Report:')
    st.text(classification_report(y_test, y_pred))


# Feature Importance Expander
with st.expander('🏆 Feature Importance'):
    st.subheader('Feature Importance')

    # Get feature importances
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    feature_names = X.columns[indices]

    # Create a figure and axis explicitly
    fig, ax = plt.subplots()
    ax.bar(range(X.shape[1]), importances[indices], align='center')
    ax.set_title('Feature Importances')
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_xlim([-1, X.shape[1]])

    # Display the plot in Streamlit
    st.pyplot(fig)

# Generate predictions probabilities for the positive class
# Probabilities for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)  # Compute the area under the ROC curve

# Create the plot within the expander
with st.expander('📈 ROC Curve'):
    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2,
            label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')

    # Display plot in Streamlit
    st.pyplot(fig)


# Lasso



