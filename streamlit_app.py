# System and I/O utilities
import os
from io import StringIO

# Core libraries
import numpy as np
import pandas as pd
import scipy.stats as stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit for app interface
import streamlit as st

# Scikit-learn: Model selection, preprocessing, and metrics
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, label_binarize
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

# Scikit-learn: Models and feature selection
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, 
    GradientBoostingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, Lasso, LassoCV, RidgeClassifier, ElasticNet
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE, SelectKBest, f_classif, chi2, VarianceThreshold, SelectFromModel
)

# Boruta for feature selection
from boruta import BorutaPy




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
st.write("**Authors:** Getnet B. (PhD Candidate), Bezawit B., Tadele B., "
         "Selam S., Michael A.")

st.info(
    "The primary objective of this project is to build a predictive model that calculates the Churn "
    "Risk Score for each customer based on their behavioral, demographic, and transactional data. "
    "This score will help the business identify customers who are at risk of churning, allowing the "
    "company to implement targeted retention strategies. The churn risk score will be represented on "
    "a scale from 1 (low churn risk) to 5 (high churn risk)."
)



st.header(" Step 1: Data Preprocessing Pipeline")

# Expanders for different data views
with st.expander('🔍 Dataset Information'):
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

with st.expander('🔢 Raw data (first 5 rows)'):
    st.write(df.head(5))  # Display first 5 rows of raw data


# Data Preprocessing
# 1.1 Handling Missing Data (5 Marks)


    
st.subheader("1.1 Handling Missing Data")
# Step 1: Replace specific values with NaN
missing_values = {
    'gender': ['Unknown'],
    'joined_through_referral': ['?'],
    'medium_of_operation': ['?'],
    'churn_risk_score': [-1]
}

# Replace specific values with NaN
df.replace(missing_values, np.nan, inplace=True)

# Replace -999 globally with NaN
df.replace(-999, np.nan, inplace=True)

# Step 2: Display missing data summary before imputation
with st.expander('⚠️ Missing Data Summary (Before Imputation)'):
    st.write(df.isnull().sum())

# Step 3: Identify Categorical and Continuous Columns
categorical_cols = [
    'gender', 'region_category', 'joined_through_referral', 
    'preferred_offer_types', 'medium_of_operation'
]

continuous_cols = [
    'points_in_wallet', 'churn_risk_score', 'days_since_last_login'
]

# Step 4: Impute Missing Values for Categorical Columns (with Mode)
for col in categorical_cols:
    mode_value = df[col].mode()[0]  # Get the most frequent value
    df[col].fillna(mode_value, inplace=True)

# Step 5: Impute Missing Values for Continuous Columns (with Median)
for col in continuous_cols:
    median_value = df[col].median()  # Get the median value
    df[col].fillna(median_value, inplace=True)

# Step 6: Display missing data summary after imputation
with st.expander('✅ Missing Data Summary (After Imputation)'):
    st.write(df.isnull().sum())



st.subheader("1.2 Data Type Correction")
## 1.2 Data Type Correction
# Step 1: Convert 'joining_date' and 'last_visit_time' to datetime format
df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], errors='coerce')
df['churn_risk_score'] = df['churn_risk_score'].astype('category')

# Step 2: Convert 'avg_frequency_login_days' from object to numeric
df['avg_frequency_login_days'] = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce')

# Step 3: Capture data types after correction
data_types_after_correction = df.dtypes

# Expander: Display data types after correction
with st.expander('🛠️ Data Types After Correction'):
    st.write(data_types_after_correction)



st.subheader("1.3 Encoding Categorical Variables")
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


st.subheader("1.4 Outlier Detection & Handling")
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



st.subheader("1.5 Feature Engineering")
# Feature Engineering
df['recency'] = df['days_since_last_login']
df['engagement_score'] = (df['avg_time_spent'] * 0.5 +
                          (1 / (df['days_since_last_login'] + 1)) * 0.5)
df['churn_history'] = df['past_complaint'].apply(
    lambda x: 1 if x == 'Yes' else 0)
df['points_utilization_rate'] = df['points_in_wallet'] / \
    (df['points_in_wallet'] + 1)
df['customer_tenure'] = (pd.to_datetime('today') - df['joining_date']).dt.days
# df['is_active'] = df['days_since_last_login'].apply(
#     lambda x: 1 if x <= 30 else 0)
df['login_frequency'] = (30 / df['days_since_last_login']
                         ).replace([float('inf'), -float('inf')], 0)
df['avg_engagement_score'] = df['avg_time_spent'] / \
    (df['days_since_last_login'] + 1)


def tenure_category(tenure):
    if tenure < 30:
        return 'New'
    elif tenure < 90:
        return 'Established'
    else:
        return 'Loyal'


df['tenure_category'] = df['customer_tenure'].apply(tenure_category)

# Update to correctly calculate offer_responsiveness
df['offer_responsiveness'] = df.apply(lambda row: 1 if any(
    offer in row['used_special_discount'] for offer in row['preferred_offer_types']) else 0, axis=1)

# Streamlit expander to show newly created columns/features
with st.expander("Show Newly Created Features"):
    st.dataframe(df[[
        'recency',
        'engagement_score',
        'churn_history',
        'points_utilization_rate',
        'customer_tenure',
        'login_frequency',
        'avg_engagement_score',
        'tenure_category',
        'offer_responsiveness'
    ]].head())


st.header("Step 2: Exploratory Data Analysis (EDA)")
st.subheader("2.1 Statistical Summaries")

# Selecting numerical columns based on the provided DataFrame structure
numerical_columns = [
    'age',
    'days_since_last_login',
    'avg_time_spent',
    'avg_transaction_value',
    'points_in_wallet',
    'customer_tenure',
    'login_frequency',
    'avg_engagement_score',
    'avg_frequency_login_days',
    'recency',
    'engagement_score',
    'points_utilization_rate',
    'churn_history',
    'offer_responsiveness'
]

# Calculate mean, median, and standard deviation
summary_stats = {
    'Mean': df[numerical_columns].mean(),
    'Median': df[numerical_columns].median(),
    'Standard Deviation': df[numerical_columns].std()
}

# Convert to DataFrame for better readability
summary_stats_df = pd.DataFrame(summary_stats)

# Streamlit expander to show summary statistics
with st.expander("Show Statistical Summaries"):
    st.dataframe(summary_stats_df)


st.subheader("2.2 Visualizations")
# Set the style of seaborn
sns.set(style="whitegrid")

# Function to plot data
def plot_data(df, columns, plot_type='count'):
    num_columns = 3 if plot_type != 'bar' else 4
    num_rows = (len(columns) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for idx, column in enumerate(columns):
        if plot_type == 'count':
            sns.countplot(df[column], ax=axes[idx], order=df[column].value_counts().index)
            axes[idx].set_title(f'Count of {column}')
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
        elif plot_type == 'hist':
            sns.histplot(df[column], bins=30, kde=True, ax=axes[idx])
            axes[idx].set_title(f'Histogram of {column}')
            axes[idx].set_xlabel(column)
            axes[idx].set_ylabel('Frequency')
        elif plot_type == 'box':
            sns.boxplot(y=df[column], ax=axes[idx])
            axes[idx].set_title(f'Box Plot of {column}')

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)  # Use Streamlit's pyplot

# List of categorical columns
categorical_columns = [
    'gender', 
    'region_category', 
    'membership_category', 
    'preferred_offer_types', 
    'tenure_category', 
    'joined_through_referral', 
    'used_special_discount', 
    'offer_application_preference', 
    'past_complaint', 
    'complaint_status', 
    'churn_risk_score'
]

# List of numerical columns
numerical_columns = [
    'age', 
    'avg_time_spent', 
    'avg_transaction_value', 
    'avg_frequency_login_days',
    'points_in_wallet', 
    'customer_tenure', 
    'login_frequency', 
    'avg_engagement_score', 
    'recency', 
    'engagement_score', 
    'churn_history', 
    'points_utilization_rate', 
    'offer_responsiveness'
]

# Create a single expander for all visualizations
with st.expander("Data Visualizations"):
    # Categorical Visualizations
    plot_data(df, categorical_columns, plot_type='count')

    ## Numerical Visualizations: Histograms
    # plot_data(df, numerical_columns, plot_type='hist')

    # Numerical Visualizations: Box plots
    plot_data(df, numerical_columns, plot_type='box')

    ## Pair Plot
    # fig = sns.pairplot(df[numerical_columns])
    # st.pyplot(fig)

    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap of Numerical Features')
    st.pyplot(plt)  # Pass the current plt figure to st.pyplot



# Treating Outlier and handling outlier for newly computed feature
new_features = [
    'customer_tenure', 'login_frequency', 'avg_engagement_score', 
    'recency', 'engagement_score', 'churn_history', 
    'points_utilization_rate', 'offer_responsiveness'
]

# Store a copy of original data for comparison (before handling)
df_new_before_handling = df[new_features].copy()

# Apply IQR method to handle outliers
for feature in new_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Handle outliers by clipping values
    df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

# Create a figure with 8 rows, 2 columns (side-by-side before & after for each feature)
fig, axs = plt.subplots(len(new_features), 2, figsize=(20, 40))

# Plot side-by-side boxplots for each feature
for i, feature in enumerate(new_features):
    # Boxplot (Before Handling)
    sns.boxplot(x=df_new_before_handling[feature], ax=axs[i, 0])
    axs[i, 0].set_title(f'{feature.replace("_", " ").title()} (Before Handling)')

    # Boxplot (After Handling)
    sns.boxplot(x=df[feature], ax=axs[i, 1])
    axs[i, 1].set_title(f'{feature.replace("_", " ").title()} (After Handling)')

    # Adjust y-axis scale to match between the two plots for fair comparison
    axs[i, 1].set_ylim(axs[i, 0].get_ylim())

# Adjust layout for better visualization
plt.tight_layout()

# Streamlit: Display the side-by-side boxplots in an expander
with st.expander('📊 Side-by-Side Boxplots for Outlier Visualization and Handling (New Features)'):
    st.pyplot(fig)
    



# Automatically detect numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

with st.expander("Show Correlation Heatmap"):
    filtered_numerical_columns = [col for col in numerical_columns if col != 'is_active']
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Calculate the correlation matrix using the filtered columns
    correlation_matrix = df[filtered_numerical_columns].corr()
    
    # Create the heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    
    # Show the plot in Streamlit
    st.pyplot(plt)



# Create a new column 'churn_status' based on churn history
df['churn_status'] = df['churn_history'].apply(lambda x: 1 if x > 0 else 0)

# Perform segmentation analysis
segmentation_analysis = (
    df.groupby(['membership_category', 'region_category', 'gender'])
    .agg(
        churn_count=('churn_status', 'sum'),
        total_customers=('customer_id', 'count'),
        churn_rate=('churn_status', 'mean')
    )
    .reset_index()
)


st.subheader("2.3 Customer Segmentation Analysis")
with st.expander("Show Segmentation Analysis"):
    st.write("**Segmentation Analysis Data**")
    st.dataframe(segmentation_analysis)

    # Optional: Display summary statistics or insights
    total_customers = segmentation_analysis['total_customers'].sum()
    avg_churn_rate = segmentation_analysis['churn_rate'].mean()
    
    st.write(f"**Total Customers:** {total_customers}")
    st.write(f"**Average Churn Rate:** {avg_churn_rate:.2%}")



st.header("Step 3: Feature Selection and Data Splitting")
st.subheader("3.1 Feature Selection")

# Step 1: Rearrange the churn_risk_score column
churn_risk_score = df.pop('churn_risk_score')  # Remove the column
df['churn_risk_score'] = churn_risk_score  # Append it to the end

# Step 2: Display raw data
with st.expander('🔢 Raw data (first 5 rows) including newly computed features before splitting'):
    st.write(df.head(5))  # Display first 5 rows of raw data

# Step 3: Prepare X (Features)
X = df.drop(columns=['customer_id', 'Name', 'security_no', 'referral_id']).iloc[:, :-1]  # Drop unnecessary columns

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

# Step 4: Check for missing values
missing_values = X[numerical_cols].isnull().sum()
if missing_values.any():
    # st.warning(f"The following columns have missing values:\n{missing_values[missing_values > 0]}")
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

# Step 5: One-Hot Encode Categorical Columns (excluding numerical columns)
X_encoded = pd.get_dummies(X[categorical_cols], drop_first=True)  # One-hot encode categorical variables

# Step 6: Normalize Numeric Features
X_numeric = X[numerical_cols]  # Select numeric columns for normalization

# Normalize the numeric features
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)

# Combine normalized features with one-hot encoded columns
X_final = pd.concat([X_normalized, X_encoded], axis=1)

with st.expander('🧩 X (Features) (first 5 rows) - Normalized and One-Hot Encoded'):
    st.write(X_final.head(5))  # Display first 5 rows of final features

# Step 7: Prepare Y (Target variable)
y = df.iloc[:, -1]  # Extract the target variable

with st.expander('🎯 Y (Target variable) (first 5 rows)'):
    st.write(y.head(5).reset_index(drop=True))  # Display the first 5 rows of the target variable





# Feature Selection Steps

# Step 1: Drop unnecessary columns
X = df.drop(columns=['customer_id', 'Name', 'security_no', 'referral_id'])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

# Handle missing values in numerical columns
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X[categorical_cols], drop_first=True)

# Normalize numeric columns
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)

# Combine encoded and normalized features
X_final = pd.concat([X_normalized, X_encoded], axis=1)

# Target variable
y = df['churn_risk_score']

# Step 2: Apply Chi-Square Feature Selection
chi2_selector = SelectKBest(chi2, k='all')
X_kbest = chi2_selector.fit_transform(X_final, y)

# Get Chi-Square scores for each feature
chi2_scores = pd.Series(chi2_selector.scores_, index=X_final.columns).sort_values(ascending=False)

# Step 3: Train/Test Split with X_kbest
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Step 5: Display Chi-Square Scores, Feature Importances, and Predictions in One Expander
with st.expander("🔍 Feature Importance and Predictions"):
    # Display Chi-Square Feature Scores
    st.write("Chi-Square Feature Scores (sorted):")
    st.write(chi2_scores)

    # Display Random Forest Feature Importances
    rf_feature_importance = pd.Series(rf.feature_importances_, index=X_final.columns).sort_values(ascending=False)
    st.write("Random Forest Feature Importances (sorted):")
    
    # Plotting Random Forest Feature Importances
    fig_rf, ax = plt.subplots(figsize=(10, 12))
    rf_feature_importance.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title("Random Forest Feature Importances")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()  # Highest importance on top
    st.pyplot(fig_rf)

    # Display Predictions (first 5 rows)
    y_pred = rf.predict(X_test)
    st.write("Random Forest Predictions (first 5):")
    st.write(pd.DataFrame({'Actual': y_test.values[:5], 'Predicted': y_pred[:5]}).reset_index(drop=True))






st.subheader("3.2 Data Splitting")
# Define split ratios
ratios = [0.1, 0.15, 0.2, 0.25, 0.3]
datasets = {}

# Create directories for saving datasets
output_dir = 'datasets'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Loop through each ratio to split and save datasets
for ratio in ratios:
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=ratio, stratify=y, random_state=42)

    # Create DataFrames for training and testing sets
    train_df = pd.DataFrame(X_train)
    train_df['churn_risk_score'] = y_train.values
    test_df = pd.DataFrame(X_test)
    test_df['churn_risk_score'] = y_test.values

    # Save datasets to CSV files in the created directory
    train_file = os.path.join(output_dir, f'train_set_{int((1-ratio)*100)}.csv')
    test_file = os.path.join(output_dir, f'test_set_{int(ratio*100)}.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    # Store datasets in the dictionary for displaying
    datasets[f'Train set {int((1-ratio)*100)}%'] = train_df
    datasets[f'Test set {int(ratio*100)}%'] = test_df

# Initialize the selected dataset before displaying
selected_dataset = list(datasets.keys())[0]  # Default selection for training set

# Display both training and testing datasets in one expander
with st.expander('🧪 Dataset Previews (Train Vs Test)'):
    # Allow user to select a dataset to view
    selected_dataset = st.selectbox('Select a training dataset here to display the test data automatically displayed:', list(datasets.keys()))
    
    # Display the selected dataset
    st.write(f"**{selected_dataset} Preview:**")
    st.write(datasets[selected_dataset].head(5))
    
    # Display the corresponding test dataset if a training dataset is selected
    if 'Train' in selected_dataset:
        test_set_key = f'Test set {100 - int(selected_dataset.split()[2][:-1])}%'
        st.write(f"**{test_set_key} Preview:**")
        st.write(datasets[test_set_key].head(5))



# Initialize the expander
with st.expander("⚙️ Data Splitting Comparison", expanded=False):
    # Define the different ratios to test
    ratios = [0.1, 0.15, 0.2, 0.25, 0.3]
    results = []

    # Iterate over each ratio to train and evaluate the model
    for ratio in ratios:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=ratio, random_state=42
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Binarize labels only if multi-class
        if len(np.unique(y)) > 2:
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))
            auc_roc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
        else:
            auc_roc = roc_auc_score(y_test, y_prob[:, 1])

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Store the results
        results.append({
            'Test Size': ratio,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc
        })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    # Display the DataFrame in Streamlit
    st.write("### Model Performance Table")
    st.dataframe(results_df.style.format({
        'Accuracy': '{:.6f}',
        'Precision': '{:.6f}',
        'Recall': '{:.6f}',
        'F1-Score': '{:.6f}',
        'AUC-ROC': '{:.6f}'
    }))

    # Plot the metrics across different split ratios
    st.write("### Performance Metrics Chart")
    st.line_chart(results_df.set_index('Test Size')[['Accuracy', 'F1-Score', 'AUC-ROC']])

    # Find the best split based on F1-Score
    best_split = results_df.loc[results_df['F1-Score'].idxmax()]
    st.write(
        f"**Best Split Ratio:** {best_split['Test Size']} "
        f"(F1-Score: {best_split['F1-Score']:.2f})"
    )





st.header("Step 4: Model Building")
st.subheader("4.1 Algorithm Selection, Model Training and Performance Metrics")

# Provide a hint for the best split ratio
# st.info("💡 Hint: The best split ratio is **25% test size / 75% train size** for this dataset.")

# Initialize the expander
with st.expander("⚙️ View Model Performance Comparison Across ML Models", expanded=False):

    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(probability=True) 
    }

    # Define the best split ratio
    best_ratio = 0.25

    # Split the data into train and test sets based on the best ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=best_ratio, random_state=42
    )

    results = []

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Binarize labels if multi-class
        if len(np.unique(y)) > 2:
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))
            auc_roc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
        else:
            auc_roc = roc_auc_score(y_test, y_prob[:, 1])

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Store the results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc
        })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    # Display the DataFrame
    st.write("### Model Performance Table")
    st.dataframe(results_df.style.format({
        'Accuracy': '{:.6f}',
        'Precision': '{:.6f}',
        'Recall': '{:.6f}',
        'F1-Score': '{:.6f}',
        'AUC-ROC': '{:.6f}'
    }))

    # Plot the metrics for comparison
    st.write("### Performance Metrics Chart")
    st.bar_chart(results_df.set_index('Model')[['Accuracy', 'F1-Score', 'AUC-ROC']])

    # Find the best model based on F1-Score
    best_model = results_df.loc[results_df['F1-Score'].idxmax()]
    st.write(
        f"**Best Model:** {best_model['Model']} "
        f"(F1-Score: {best_model['F1-Score']:.2f})"
    )




st.subheader("4.2 Hyperparameter Tuning with Gradient Boosting Classifier")
st.info(
    "📌 **Remark:**\n"
    "Due to the limitations of the current system's performance, the hyperparameter tuning code has been commented out. "
    "We tested this tuning process on a smaller dataset, and it performed successfully."
)
# # Initialize the expander
# with st.expander("⚙️ Hyperparameter Tuning for Gradient Boosting", expanded=False):
#     # Split the dataset again with the best ratio
#     X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42)

#     # Define the model
#     gb_model = GradientBoostingClassifier()

#     # Define the hyperparameters to tune
#     param_grid = {
#         'n_estimators': [50, 100, 150],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'max_depth': [3, 4, 5],
#         'min_samples_split': [2, 5, 10]
#     }

#     # Perform Grid Search
#     grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, scoring='f1_weighted', cv=5, verbose=1)
#     grid_search.fit(X_train, y_train)

#     # Get the best model and parameters
#     best_model = grid_search.best_estimator_
#     best_params = grid_search.best_params_

#     # Make predictions on the test set
#     y_pred = best_model.predict(X_test)

#     # Display results
#     st.write("### Best Hyperparameters")
#     st.json(best_params)

#     st.write("### Classification Report")
#     st.text(classification_report(y_test, y_pred))

#     st.write("### Confusion Matrix")
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     st.write(conf_matrix)

#     # Visualize the confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     st.pyplot(plt)




# Initialize the expander
with st.expander("⚙️ K-Fold Cross-Validation", expanded=False):
    # Split the dataset with the best ratio
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42)

    # Define the model
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_split=2)

    # Perform k-fold cross-validation
    k = 5  # We can adjust k as needed
    cv_scores = cross_val_score(gb_model, X_train, y_train, cv=k, scoring='f1_weighted')

    # Display results
    st.write("### Cross-Validation Results")
    st.write(f"F1-Score for each fold: {cv_scores}")
    st.write(f"Mean F1-Score: {cv_scores.mean():.4f}")
    st.write(f"Standard Deviation: {cv_scores.std():.4f}")

    # Optionally, we can visualize the scores
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, k + 1), cv_scores, marker='o', linestyle='-', color='b')
    plt.title('K-Fold Cross-Validation F1-Scores')
    plt.xlabel('Fold Number')
    plt.ylabel('F1-Score')
    plt.xticks(range(1, k + 1))
    plt.grid()
    st.pyplot(plt)






# Function to convert minutes to HH:MM:SS format
def minutes_to_time(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{int(hours):02}:{int(mins):02}:00"

# Sidebar for top input features
with st.sidebar:
    st.header('Top Input Features')

    # Input fields for the top features
    membership_category = st.selectbox('Membership Category', [
        'Platinum Membership', 'Premium Membership', 'No Membership', 
        'Gold Membership', 'Silver Membership', 'Basic Membership'], index=0)

    # Slider for points utilization rate (assumed as a percentage)
    points_utilization_rate = st.slider('Points Utilization Rate (%)', 0.0, 100.0, 50.0, step=0.1)
    
    # Slider for points in wallet
    points_in_wallet = st.slider('Points in Wallet', -549.36, 1755.09, 0.0, step=0.01)
    
    # Slider for average transaction value
    avg_transaction_value = st.slider('Average Transaction Value', 806.22, 99805.52, 5000.0, step=0.01)
    
    # Slider for customer tenure (assumed in days here)
    customer_tenure = st.slider('Customer Tenure (in days)', 0, 365 * 5, 365, step=1)
    
    # Slider for average frequency of login days
    avg_frequency_login_days = st.slider('Average Frequency of Login Days', -43.65, 73.06, 0.0, step=0.01)
    
    # Slider for engagement score
    engagement_score = st.slider('Engagement Score', 0, 100, 50, step=1)

    # Input for age
    age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
    
    # Slider for average engagement score
    avg_engagement_score = st.slider('Average Engagement Score', 0.0, 100.0, 50.0, step=0.1)
    
    # Slider for average time spent (in minutes)
    avg_time_spent = st.slider('Average Time Spent', -2096.58, 2732.70, 0.0, step=0.01)
    
    # Selectbox for feedback
    feedback = st.selectbox('Feedback', [
        'Products always in Stock', 'Quality Customer Care', 'Poor Website', 
        'No reason specified', 'Poor Product Quality', 'Too many ads', 
        'User Friendly Website', 'Poor Customer Service', 'Reasonable Price'], index=0)
    
    # Slider for recency (days since last interaction)
    recency = st.slider('Recency (Days Since Last Interaction)', 0, 365, 30, step=1)
    
    # Slider for days since last login
    days_since_last_login = st.slider('Days Since Last Login', 1, 365, 13, step=1)
    
    # Slider for login frequency
    login_frequency = st.slider('Login Frequency (per week)', 0.0, 7.0, 3.5, step=0.1)



# Example button to confirm input submission
if st.button('Submit'):
    st.write('Input features submitted successfully!')

# Split the data (make sure to define X_final and y before this)
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42)

# Cache the model training
@st.cache_resource
def train_model(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model

# Train the model with a spinner
with st.spinner('Training the model...'):
    model = train_model(X_train, y_train)

# # Prepare input data function
# def encode_inputs(X_columns):
#     input_values = [st.number_input(f'{col}', key=col) for col in X_columns]
#     input_data = np.array(input_values).reshape(1, -1)
#     return input_data

# # Get the columns from our final dataset
# X_columns = X_final.columns

# # Prepare the input data
# input_data = encode_inputs(X_columns)

# Predict and display results when button is clicked
if st.button('👉 Click Me to Calculate Churn Risk Score'):
    with st.spinner('Calculating churn risk...'):
        # Predict probabilities for each class
        prediction_proba = model.predict_proba(input_data)[0]

        # Displaying as a bar chart
        st.bar_chart(prediction_proba, use_container_width=True)

        # Display the predicted category with the highest probability
        predicted_category = np.argmax(prediction_proba) + 1  # categories start from 1
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



