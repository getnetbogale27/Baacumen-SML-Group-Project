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
from io import StringIO 


@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/getnetbogale27/Baacumen-SML-Group-Project/main/Dataset/train.csv'
    df = pd.read_csv(url)
    # Rename dataset to 'ML-Data'
    ML_Data = df.copy()  
    # Save the renamed dataset to a CSV file
    ML_Data.to_csv('ML-Data.csv', index=False)
    return ML_Data

# Load the data
df = load_data()


#Step 2
# Streamlit App Setup
# Streamlit title and info
st.title('🤖 Supervised ML App')
# List of Authors
st.write("**Authors:** Getnet B. (PhD Candidate), Bezawit Belachew (Dr.), Tadele, "
         "Selamawit, Michiakel")

st.info(
    "The primary objective of this project is to build a predictive model that calculates the Churn "
    "Risk Score for each customer based on their behavioral, demographic, and transactional data. "
    "This score will help the business identify customers who are at risk of churning, allowing the "
    "company to implement targeted retention strategies. The churn risk score will be represented on "
    "a scale from 1 (low churn risk) to 5 (high churn risk)."
)





# Expanders for different data views
with st.expander('🔍 Dataset Information'):
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

with st.expander('🔢 Raw data (first 5 rows)'):
    st.write(df.head(5))  # Display first 5 rows of raw data

with st.expander('🧩 X (independent variables) (first 5 rows)'):
    X_raw = df.iloc[:, 2:-1]  # Adjusted to start from the 3rd column (index 2)
    st.write(X_raw.head(5)) 

with st.expander('🎯 Y (dependent variable) (first 5 rows)'):
    y_raw = df.iloc[:, -1]  
    st.write(y_raw.head(5).reset_index(drop=True))


## Data Preprocessing
## 1.1 Handling Missing Data (5 Marks)

# Define specific values to replace with NaN
missing_values = {
    'gender': ['Unknown'],
    'joined_through_referral': ['?'],
    'medium_of_operation': ['?'],
    'churn_risk_score': [-1]
}

# Replace these specified values with NaN
df.replace(missing_values, np.nan, inplace=True)

# Capture the missing data summary
missing_data_summary = df.isnull().sum()

# Create an expander to show missing data summary
with st.expander('⚠️ Missing Data Summary (Before Imputation)'):
    st.write(missing_data_summary)
    # st.write(data.isnull().sum())


# Treat for missing values
# Step 1: Define specific values to replace with NaN
# Allready we done previously

# Step 2: Handle Categorical Columns
categorical_cols = [
    'gender', 'region_category', 'joined_through_referral', 
    'preferred_offer_types', 'medium_of_operation'
]

for col in categorical_cols:
    if col in ['joined_through_referral', 'medium_of_operation']:
        # Impute with 'Unknown'
        df[col].fillna('Unknown', inplace=True)
    else:
        # Impute with mode (most frequent value)
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)

# Step 3: Handle Numerical Columns
df['points_in_wallet'].fillna(df['points_in_wallet'].median(), inplace=True)
df['churn_risk_score'].fillna(df['churn_risk_score'].median(), inplace=True)

# Expander 2: Show missing data summary after imputation
with st.expander('✅ Missing Data Summary (After Imputation)'):
    st.write(df.isnull().sum())


## 1.2 Data Type Correction
# Step 1: Convert 'joining_date' and 'last_visit_time' to datetime format
df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], errors='coerce')
df['churn_risk_score'] = df['churn_risk_score'].astype('category')

# Step 2: Capture data types after correction
data_types_after_correction = df.dtypes

# Expander: Display data types after correction
with st.expander('🛠️ Data Types After Correction'):
    st.write(data_types_after_correction)


# 1.3 Encoding Categorical Variables
# Identify categorical columns for One-Hot Encoding
categorical_columns_one_hot = [
    'gender', 'region_category', 'membership_category', 'joined_through_referral',
    'preferred_offer_types', 'medium_of_operation', 'used_special_discount',
    'offer_application_preference', 'past_complaint', 'complaint_status', 'feedback'
]

# Perform One-Hot Encoding
data_one_hot_encoded = pd.get_dummies(df, columns=categorical_columns_one_hot, drop_first=True)

# Expander: Display the first few rows of the new DataFrame
with st.expander('🔠 One-Hot Encoded Data Sample'):
    st.write(data_one_hot_encoded.head())


# Function to detect and handle outliers using IQR
def detect_outliers_iqr(df, column):
    if column not in df.columns:
        print(f"Column {column} does not exist in the DataFrame.")
        return df

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # Print detected outliers in the console (optional)
    print(f"Detected outliers in {column}:\n", outliers)
    
    # Handling outliers by capping
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df  # Return the modified DataFrame

# Applying IQR method to specific columns
columns_to_check = ['age', 'avg_time_spent', 'avg_transaction_value', 'points_in_wallet']
df_before_handling = df.copy()  # Make a copy for comparison

for col in columns_to_check:
    df = detect_outliers_iqr(df, col)

# Set up the matplotlib figure for the first set of boxplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Create boxplots for Age
sns.boxplot(x=df_before_handling['age'], ax=axs[0, 0])
axs[0, 0].set_title('Boxplot of Age (Before Handling)')

sns.boxplot(x=df['age'], ax=axs[0, 1])
axs[0, 1].set_title('Boxplot of Age (After Handling)')

# Create boxplots for Average Time Spent
sns.boxplot(x=df_before_handling['avg_time_spent'], ax=axs[1, 0])
axs[1, 0].set_title('Boxplot of Average Time Spent (Before Handling)')

sns.boxplot(x=df['avg_time_spent'], ax=axs[1, 1])
axs[1, 1].set_title('Boxplot of Average Time Spent (After Handling)')

plt.tight_layout()

# Create expanders to display the data in Streamlit
with st.expander('📊 Data Before Handling Outliers'):
    st.write(df_before_handling.describe())

with st.expander('📊 Data After Handling Outliers'):
    st.write(df.describe())

# Combine boxplots for Age, Average Time Spent, Average Transaction Value, and Points in Wallet
fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))

# Boxplots for Average Transaction Value
sns.boxplot(x=df_before_handling['avg_transaction_value'], ax=axs2[0, 0])
axs2[0, 0].set_title('Boxplot of Average Transaction Value (Before Handling)')

sns.boxplot(x=df['avg_transaction_value'], ax=axs2[0, 1])
axs2[0, 1].set_title('Boxplot of Average Transaction Value (After Handling)')

# Boxplots for Points in Wallet
sns.boxplot(x=df_before_handling['points_in_wallet'], ax=axs2[1, 0])
axs2[1, 0].set_title('Boxplot of Points in Wallet (Before Handling)')

sns.boxplot(x=df['points_in_wallet'], ax=axs2[1, 1])
axs2[1, 1].set_title('Boxplot of Points in Wallet (After Handling)')

plt.tight_layout()

# Create a single expander for all boxplots
with st.expander('📊 Boxplots for Outlier Visualization and Handling'):
    st.pyplot(fig)
    st.pyplot(fig2)


# 1.5 Feature Engineering
# Feature Engineering
df['recency'] = df['days_since_last_login']  # Days since last login

# Calculate Monetary Value
df['monetary_value'] = df['avg_transaction_value'] * df['transaction_frequency']  # Total monetary value
# Calculate Average Order Value (AOV)
df['average_order_value'] = df['monetary_value'] / (df['transaction_frequency'] + 1)  # AOV safeguard

# Calculate Engagement Score
df['engagement_score'] = (df['avg_time_spent'] * 0.5 + (1 / (df['days_since_last_login'] + 1)) * 0.5)

# Churn History
df['churn_history'] = df['past_complaint'].apply(lambda x: 1 if x == 'Yes' else 0)

# Points Utilization Rate
df['points_utilization_rate'] = df['points_in_wallet'] / (df['points_in_wallet'] + 1)

# Customer Tenure in Days
df['customer_tenure'] = (pd.to_datetime('today') - df['joining_date']).dt.days

# Is Active
df['is_active'] = df['days_since_last_login'].apply(lambda x: 1 if x <= 30 else 0)

# Login Frequency
df['login_frequency'] = (30 / df['days_since_last_login']).replace([float('inf'), -float('inf')], 0)

# Average Engagement Score
df['avg_engagement_score'] = df['avg_time_spent'] / (df['days_since_last_login'] + 1)

# Tenure-Based Segmentation
def tenure_category(tenure):
    if tenure < 30:
        return 'New'
    elif tenure < 90:
        return 'Established'
    else:
        return 'Loyal'

df['tenure_category'] = df['customer_tenure'].apply(tenure_category)

# Offer Responsiveness
df['offer_responsiveness'] = df.apply(lambda x: 1 if x['preferred_offer_types'] in x['used_special_discount'] else 0, axis=1)

# Display the newly created features using Streamlit expander
with st.expander("Newly Created Features", expanded=True):
    st.write(df[[
        'recency', 
        'engagement_score', 
        'churn_history', 
        'points_utilization_rate', 
        'customer_tenure', 
        'is_active', 
        'login_frequency', 
        'avg_engagement_score', 
        'tenure_category', 
        'offer_responsiveness'
    ]].head())











# function to convert minutes to HH:MM:SS format
def minutes_to_time(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{int(hours):02}:{int(mins):02}:00"

# Sidebar for input features
with st.sidebar:
    st.header('Input Features')

    # Input fields
    gender = st.selectbox('Gender', ['F', 'M'], index=0)
    region_category = st.selectbox('Region Category', ['Village', 'City', 'Town'], index=0)
    membership_category = st.selectbox('Membership Category', [
        'Platinum Membership', 'Premium Membership', 'No Membership', 
        'Gold Membership', 'Silver Membership', 'Basic Membership'], index=0)

    joining_date = st.date_input(
        'Joining Date', value=pd.to_datetime('2015-01-01'),
        min_value=pd.to_datetime('2015-01-01'),
        max_value=pd.to_datetime('2017-12-31'))

    joined_through_referral = st.selectbox('Joined through Referral', ['Yes', 'No'], index=0)
    preferred_offer_types = st.selectbox('Preferred Offer Types', [
        'Gift Vouchers/Coupons', 'Credit/Debit Card Offers', 'Without Offers'], index=0)
    
    medium_of_operation = st.selectbox('Medium of Operation', ['Both', 'Desktop', 'Smartphone'], index=0)
    internet_option = st.selectbox('Internet Option', ['Wi-Fi', 'Mobile_Data', 'Fiber_Optic'], index=0)

    # Slider for last visit time (in minutes)
    last_visit_minutes = st.slider(
        'Last Visit Time in Minute', min_value=0, max_value=24 * 60 - 1, value=12 * 60, step=1
    )
    last_visit_time = minutes_to_time(last_visit_minutes)

    days_since_last_login = st.slider('Days Since Last Login', 1, 26, 13, step=1)
    avg_time_spent = st.slider('Average Time Spent', -2096.58, 2732.70, 0.0, step=0.01)
    avg_transaction_value = st.slider('Average Transaction Value', 806.22, 99805.52, 5000.0, step=0.01)
    avg_frequency_login_days = st.slider('Average Frequency of Login Days', -43.65, 73.06, 0.0, step=0.01)
    points_in_wallet = st.slider('Points in Wallet', -549.36, 1755.09, 0.0, step=0.01)

    used_special_discount = st.selectbox('Used Special Discount', ['Yes', 'No'], index=0)
    offer_application_preference = st.selectbox('Offer Application Preference', ['Yes', 'No'], index=0)
    past_complaint = st.selectbox('Past Complaint', ['Yes', 'No'], index=0)

    complaint_status = st.selectbox('Complaint Status', [
        'Not Applicable', 'Solved', 'Solved in Follow-up', 'Unsolved', 'No Information Available'], index=0)

    feedback = st.selectbox('Feedback', [
        'Products always in Stock', 'Quality Customer Care', 'Poor Website', 
        'No reason specified', 'Poor Product Quality', 'Too many ads', 
        'User Friendly Website', 'Poor Customer Service', 'Reasonable Price'], index=0)

# Example button to confirm input submission
if st.button('Submit'):
    st.write('Input features submitted successfully!')


















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



