import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from scipy import stats  
import warnings  

warnings.filterwarnings("ignore")

#Part1
data = pd.read_csv("Crimes_-_2001_to_Present.csv")
print(data.head())

data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
data = data.dropna(subset=["Primary Type", "Location Description", "Date"])
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Hour"] = data["Date"].dt.hour

print("Data shape:", data.shape)
print("Columns:", data.columns)

#Part2
top_crimes = data["Primary Type"].value_counts().head(10)
print("Top 10 crimes:")
print(top_crimes)
plt.figure(figsize=(10,6))
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="Set2")
plt.title("Top 10 Crime Types in Chicago")
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")
plt.show()

#Part3
crimes_per_year = data["Year"].value_counts().sort_index()
plt.figure(figsize=(12,6))
plt.plot(crimes_per_year.index, crimes_per_year.values, marker='o')
plt.title("Total Crimes Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.show()

#Part4
plt.figure(figsize=(10,5))
sns.countplot(x="Hour", data=data, palette="coolwarm")
plt.title("Crimes by Hour of the Day")
plt.xlabel("Hour (0 = midnight)")
plt.ylabel("Number of Crimes")
plt.show()

#Part5
print(data.describe())
corr = data[["Year", "Month", "Hour"]].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="Blues")
plt.title("Correlation Between Year, Month, and Hour")
plt.show()

#Part6
plt.figure(figsize=(8,5))
sns.boxplot(x=data["Hour"])
plt.title("Outliers in Crime Hours")
plt.show()

#Part7
t_stat, p_val = stats.ttest_1samp(data["Year"], 2010)

print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("Result: Crime years are significantly different from 2010")
else:
    print("Result: Crime years are not significantly different from 2010")


#Part8
crime_counts = data["Primary Type"].value_counts().head(5)

chi_stat, chi_p = stats.chisquare(crime_counts)

print("Chi-Squared Statistic:", chi_stat)
print("P-Value:", chi_p)

if chi_p < 0.05:
    print("Result: Not evenly distributed")
else:
    print("Result: May be evenly distributed")
