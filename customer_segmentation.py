# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
bank_data = pd.read_csv('bank_data.csv')

# Check for missing values and handle them accordingly
bank_data.isnull().sum()

# Check for duplicates and remove them if necessary
bank_data.drop_duplicates(inplace=True)

# Convert categorical variables to numerical format
bank_data = pd.get_dummies(bank_data)

# Normalize the numerical variables
scaler = StandardScaler()
bank_data_scaled = scaler.fit_transform(bank_data)

# Visualize the distribution of each variable
sns.pairplot(bank_data)

# Identify any outliers or anomalies
sns.boxplot(data=bank_data)

# Perform correlation analysis to identify any significant relationships between variables
corr_matrix = bank_data.corr()
sns.heatmap(corr_matrix, annot=True)

# Perform feature selection to identify the most relevant variables for customer segmentation
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=6)
selector.fit(bank_data)
selected_cols = bank_data.columns[selector.get_support(indices=True)]
bank_data = bank_data[selected_cols]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bank_data_scaled,  test_size=0.2, random_state=0)

# Select appropriate machine learning algorithms for customer segmentation, such as K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)

# Tune the hyperparameters of the chosen algorithm(s) using cross-validation techniques
kmeans.fit(X_train)

# Evaluate the performance of the algorithm(s) using metrics such as silhouette score
score = silhouette_score(X_train, kmeans.labels_, metric='euclidean')
print('Silhouette Score:', score)

# Apply the chosen algorithm(s) to the testing set to obtain the final customer segmentation
y_pred = kmeans.predict(X_test)

# Interpret the results of the customer segmentation and identify the different customer groups based on their characteristics
cluster_df = pd.DataFrame(X_test, columns=bank_data.columns)
cluster_df['Cluster'] = y_pred
cluster_df.head()

# Analyze the marketing needs and preferences of each customer group
cluster1 = cluster_df[cluster_df['Cluster'] == 0]
cluster2 = cluster_df[cluster_df['Cluster'] == 1]
cluster3 = cluster_df[cluster_df['Cluster'] == 2]

# Provide recommendations for targeted marketing campaigns for each customer group based on their needs and preferences
# Summarize the data preprocessing, exploratory data analysis, machine learning, interpretation, and recommendations
print('Summary Report:\n')
print('-'*50)
print('Data Preprocessing:\n')
print('The dataset was loaded and checked for missing values and duplicates. Categorical variables were converted to numerical format and the numerical variables were normalized using StandardScaler.')
print('\n')
print('-'*50)
print('Exploratory Data Analysis:\n')
print('The distribution of each variable was visualized and any outliers or anomalies were identified. Correlation analysis')
