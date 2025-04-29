import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('data.csv')
df

df.isnull().sum()

df['Parental_Education'] = df['Parental_Education'].replace('None','Uneducated')

df['Parental_Education'] = df['Parental_Education'].fillna('Uneducated')

df

df.isnull().sum()

df.loc[df.sample(frac=0.1, random_state=42).index, 'Test_Scores'] = np.nan
df.loc[df.sample(frac=0.05, random_state=42).index, 'Parental_Education'] = np.nan
df

df=df.drop(columns=["Unnamed: 0","Student_ID"],axis=1)

df.isnull().sum()

df.head()

df.shape

df.info()

df.describe()

df['Parental_Education'].unique()

df['Parental_Education'].value_counts()

df['Socioeconomic_Status'].unique()

df['Socioeconomic_Status'].value_counts()

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style="whitegrid")
# fig, axes = plt.subplots(3, 2, figsize=(16, 16))
# sns.histplot(df['Attendance_Rate'], kde=True, ax=axes[0, 0], color="skyblue")
# axes[0, 0].set_title("Distribution of Attendance Rate")
# sns.histplot(df['Test_Scores'], kde=True, ax=axes[0, 1], color="orange")
# axes[0, 1].set_title("Distribution of Test Scores")
# sns.histplot(df['Engagement_Score'], kde=True, ax=axes[1, 0], color="green")
# axes[1, 0].set_title("Distribution of Engagement Score")
# sns.countplot(x='Gender', data=df, ax=axes[1, 1], palette="viridis")
# axes[1, 1].set_title("Count of Gender")
# sns.countplot(x='Parental_Education', data=df, ax=axes[2, 0], palette="pastel")
# axes[2, 0].set_title("Parental Education Levels")
# axes[2, 0].tick_params(axis='x', rotation=45)
# sns.countplot(x='Socioeconomic_Status', data=df, ax=axes[2, 1], palette="cool")
# axes[2, 1].set_title("Socioeconomic Status")
# plt.tight_layout()
# plt.show()

# numeric_cols = ['Attendance_Rate', 'Test_Scores', 'Engagement_Score', 'Dropout_Risk']
# numeric_corr_matrix = df[numeric_cols].corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(numeric_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
# plt.title("Correlation Heatmap: Numeric vs Numeric")
# plt.show()

# categorical_ct = pd.crosstab(df['Gender'], df['Dropout_Risk'], normalize='index')
# plt.figure(figsize=(8, 6))
# sns.heatmap(categorical_ct, annot=True, fmt=".2f", cmap="Blues", cbar=True)
# plt.title("Proportion Heatmap: Categorical vs Categorical (Gender vs Dropout Risk)")
# plt.ylabel("Gender")
# plt.xlabel("Dropout Risk")
# plt.show()

# numeric_vs_categorical = df.groupby('Dropout_Risk')[numeric_cols[:-1]].mean()
# plt.figure(figsize=(8, 6))
# sns.heatmap(numeric_vs_categorical, annot=True, fmt=".2f", cmap="Greens", cbar=True)
# plt.title("Mean Value Heatmap: Numeric vs Categorical (Grouped by Dropout Risk)")
# plt.ylabel("Dropout Risk")
# plt.xlabel("Numeric Variables")
# plt.show()

df.isnull().sum()

Parental_Education_mode = df['Parental_Education'].mode()[0]  # Get the most frequent value
df['Parental_Education'].fillna(Parental_Education_mode, inplace=True)

df.isnull().sum()

def missing_data(df,var,mean,median,mode):
  df[var+'_mean']=df[var].fillna(mean)
  df[var+'_median']=df[var].fillna(median)
  df[var+'_mode']=df[var].fillna(mode[0])

Test_Scores_mean = df['Test_Scores'].mean()
Test_Scores_median = df['Test_Scores'].median()
Test_Scores_mode = df['Test_Scores'].mode()

missing_data(df,'Test_Scores',Test_Scores_mean,Test_Scores_median,Test_Scores_mode)

# plt.figure(figsize =(5,5))
# df['Test_Scores'].plot(kind='kde',color='r',label='Original')
# df['Test_Scores_mean'].plot(kind='kde',color='b',label='Age_mean')
# df['Test_Scores_median'].plot(kind='kde',color='g',label='Age_median')
# df['Test_Scores_mode'].plot(kind='kde',color='y',label='Age_mode')
# plt.legend(loc =0)
# plt.show()

Test_Scores_median = df['Test_Scores'].median()
df['Test_Scores'].fillna( Test_Scores_median,inplace=True)

df=df.drop(['Test_Scores_mean','Test_Scores_mode','Test_Scores'],axis=1)

df.isnull().sum()

# def visualize_outliers(df, numeric_columns):
#     plt.figure(figsize=(15, 15))
#     for i, col in enumerate(numeric_columns, start=1):
#         plt.subplot(len(numeric_columns), 2, 2 * i - 1)
#         sns.boxplot(df[col], orient='h', color='skyblue')
#         plt.title(f'Boxplot of {col}')
#     plt.tight_layout()
#     plt.show()
# numeric_columns = ["Attendance_Rate", "Test_Scores_median", "Engagement_Score"]
# visualize_outliers(df, numeric_columns)

df.head()

df.select_dtypes('object')

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
df['Parental_Education'] = ordinal_encoder.fit_transform(df[['Parental_Education']]).astype(int)
df['Socioeconomic_Status'] = ordinal_encoder.fit_transform(df[['Socioeconomic_Status']]).astype(int)




df['Gender']=df['Gender'].map({'Male':0,'Female':1}).astype(int)



X = df.drop(['Dropout_Risk'], axis = 1)
y = df['Dropout_Risk']



X.columns

X.shape

# X.to_csv('Drop.csv',index=False)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state =42)


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# id = pd.read_csv("Drop.csv")
# id.head()
import joblib

joblib.dump(dt, 'decision.pkl')

y_pred = dt.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))