# Step 1: Import necessary packages and dataset
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
    url = 'https://raw.githubusercontent.com/getnetbogale27/Child-Undernutrition-ML-app/refs/heads/master/data/ML-Data.csv?token=GHSAT0AAAAAACY5XTIM2JABHYUASMZWHMVEZYU3WTA'
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



