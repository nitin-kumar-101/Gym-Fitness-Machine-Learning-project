import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'D:\INT375_Python\Project\Predictive Analysis\data\gym_dataset_1000_rows.csv')

print("First five rows:")
print(df.head())


print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

df_clean = df.drop_duplicates()

df_clean = df_clean.reset_index(drop=True)

print("\nOriginal dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)

print("Remaining duplicates:", df_clean.duplicated().sum())

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

df_clean.to_csv('gym_dataset_cleaned.csv', index=False)

print("\nCleaned dataset saved successfully!")

# ==========================================
# STEP 3.1: DISTRIBUTION OF CALORIES BURNED
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df_clean,
    x='Calories_Burned',
    kde=True
)

plt.title('Distribution of Calories Burned')
plt.xlabel('Calories Burned')
plt.ylabel('Frequency')

plt.show()

# ==========================================
# STEP 3.2: CORRELATION HEATMAP
# ==========================================

df_numeric = df_clean.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))

sns.heatmap(
    df_numeric.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap of Health and Fitness Features')

plt.show()

