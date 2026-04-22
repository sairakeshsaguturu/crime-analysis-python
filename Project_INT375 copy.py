# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Style
sns.set_theme(style="whitegrid")

# =========================================
# 2. LOAD DATA
# =========================================
df = pd.read_csv('/Users/rakesh/Desktop/DataSet/Crime_Data_from_2020_to_2024.csv')

print("Initial Shape:", df.shape)
print(df.head())

# =========================================
# 3. DATA CLEANING
# =========================================
df['DATE OCC'] = pd.to_datetime(
    df['DATE OCC'],
    format='%m/%d/%Y %I:%M:%S %p',
    errors='coerce'
)

df = df.dropna(subset=['DATE OCC'])

df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
df['Hour'] = df['TIME OCC'].str[:2].astype(int)

df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month

# Filter 2020
df = df[df['Year'] == 2020]

print("Cleaned Shape:", df.shape)

# =========================================
# 4. BASIC EDA
# =========================================
print(df.info())
print(df.describe())
print(df.isnull().sum())

# =========================================
# 5. VISUALIZATIONS
# =========================================

# Crimes by Month
plt.figure()
sns.countplot(x='Month', hue='Month', data=df, palette="viridis", legend=False)
plt.title("Crimes by Month (2020)")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()

# Crimes by Hour
plt.figure()
sns.countplot(x='Hour', hue='Hour', data=df, palette="coolwarm", legend=False)
plt.title("Crimes by Hour")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.show()

# Top Crime Types
top_crimes = df['Crm Cd Desc'].value_counts().head(10)
plt.figure()
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="magma")
plt.title("Top 10 Crime Types")
plt.xlabel("Count")
plt.ylabel("Crime Type")
plt.show()

# Victim Gender Distribution
plt.figure()
df['Vict Sex'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['#66b3ff', '#ff9999', '#99ff99']
)
plt.title("Victim Gender Distribution")
plt.ylabel("")
plt.show()

# Top Areas (Donut Chart)
area_counts = df['AREA NAME'].value_counts().head(5)
plt.figure()
plt.pie(
    area_counts,
    labels=area_counts.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("pastel")
)
centre_circle = plt.Circle((0, 0), 0.6, fc='white')
plt.gca().add_artist(centre_circle)
plt.title("Top Areas")
plt.show()

# Histogram
plt.figure()
sns.histplot(df['Hour'], kde=True, bins=24, color='orange')
plt.title("Crime Hour Distribution")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
plt.figure()
sns.scatterplot(x='Month', y='Hour', data=df, color='green')
plt.title("Month vs Hour")
plt.xlabel("Month")
plt.ylabel("Hour")
plt.show()

# Box Plot
plt.figure()
sns.boxplot(x='Month', y='Hour', data=df, palette="Set3")
plt.title("Hour Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Hour")
plt.show()

# Top 10 Crime Areas
plt.figure()
sns.countplot(
    y='AREA NAME',
    data=df,
    order=df['AREA NAME'].value_counts().head(10).index,
    palette="cool"
)
plt.title("Top 10 Crime Areas")
plt.xlabel("Count")
plt.ylabel("Area")
plt.show()

# Heatmap
numeric_df = df.select_dtypes(include=np.number)
plt.figure()
sns.heatmap(numeric_df.corr(), annot=True, cmap="RdYlBu")
plt.title("Correlation Heatmap")
plt.show()

# =========================================
# 6. SKEWNESS
# =========================================
print("Skewness:\n", numeric_df.skew())

# =========================================
# 7. OUTLIERS
# =========================================
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((numeric_df < (Q1 - 1.5 * IQR)) |
            (numeric_df > (Q3 + 1.5 * IQR)))

print("Outliers per column:\n", outliers.sum())
print("Total rows with outliers:", outliers.any(axis=1).sum())

# =========================================
# 8. MACHINE LEARNING
# =========================================
model_df = df[['Hour']].copy()
model_df['Index'] = range(len(model_df))

X = model_df[['Index']]
y = model_df[['Hour']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Sample Predictions:\n", pred[:5])

plt.figure()
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Linear Regression (Crime Trend)")
plt.xlabel("Index")
plt.ylabel("Hour")
plt.show()

# =========================================
# 9. KEY INSIGHTS
# =========================================
print("Peak Crime Hour:", df['Hour'].value_counts().idxmax())
print("Most Common Crime:", df['Crm Cd Desc'].value_counts().idxmax())
print("Most Dangerous Area:", df['AREA NAME'].value_counts().idxmax())
