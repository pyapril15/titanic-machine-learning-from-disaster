"""
Minor Project - 2
Titanic Survival Analysis

Dataset: Titanic - Machine Learning from Disaster (Kaggle)

How to run:
python project.py

Keep titanic.csv inside the data folder.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# setting the chart style
sns.set_theme(style="whitegrid")

# charts saved here automatically
os.makedirs("charts", exist_ok=True)


def save_chart(chart_name):
    plt.tight_layout()
    plt.savefig(f"charts/{chart_name}.png", dpi=150)
    plt.close()


# -------------------- 1. LOAD THE DATASET --------------------

print("\nLoading Titanic dataset...")

df = pd.read_csv("data/titanic.csv")

print("\nDataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

# -------------------- 2. DATA CLEANING --------------------

print("\nChecking missing values:")

missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# filling missing ages with the median value
# i used median because a few unusual ages can affect the mean.
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked has only a few missing entries, so using the most common value.
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin contains too many missing values, so dropping it for now.
df.drop("Cabin", axis=1, inplace=True)

# adding a column to check how many family members travelled together
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# dividing passengers into different age groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 40, 60, 100],
    labels=["Child", "Teen", "Adult", "Middle-aged", "Senior"]
)

print("\nUpdated dataset:")
print(df.head())

print("\nBasic statistical summary:")
print(df[["Age", "Fare", "FamilySize"]].describe().round(2))

# -------------------- 3. DATA ANALYSIS --------------------

print("\n----------- SURVIVAL ANALYSIS -----------")

# overall survival percentage
survival_percentage = df["Survived"].mean() * 100

print("\nOverall survival rate: {:.1f}%".format(survival_percentage))

# survival comparison between male and female passengers
print("\nSurvival rate by gender (%):")
gender_result = df.groupby("Sex")["Survived"].mean() * 100
print(gender_result.round(1))

# checking whether passenger class affected survival
print("\nSurvival rate by passenger class (%):")
class_result = df.groupby("Pclass")["Survived"].mean() * 100
print(class_result.round(1))

# comparing survival across different age groups
print("\nSurvival rate by age group (%):")
age_result = df.groupby("AgeGroup", observed=True)["Survived"].mean() * 100
print(age_result.round(1))

# checking whether travelling with family made a difference
print("\nSurvival rate by family size (%):")
family_result = df.groupby("FamilySize")["Survived"].mean() * 100
print(family_result.round(1))

# -------------------- 4. DATA VISUALIZATION --------------------

print("\nCreating charts...")

# chart 1: overall survival count
# this pie chart shows how many passengers survived.

survival_count = df["Survived"].map({
    0: "Did not survive",
    1: "Survived"
}).value_counts()

survival_count.plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["#e07a5f", "#81b29a"],
    startangle=90,
    ylabel=""
)

plt.title("Overall Survival")
save_chart("1_survival_pie")

# chart 2: survival rate by gender
# comparing male and female passengers.

sns.barplot(
    data=df,
    x="Sex",
    y="Survived",
    hue="Sex",
    palette="Set2",
    errorbar=None,
    legend=False
)

plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
save_chart("2_survival_by_sex")

# chart 3: survival rate by passenger class
# let's see if travelling in first class made a difference.

sns.barplot(
    data=df,
    x="Pclass",
    y="Survived",
    hue="Pclass",
    palette="Blues_d",
    errorbar=None,
    legend=False
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Class (1 = Upper, 3 = Lower)")
plt.ylabel("Survival Rate")
save_chart("3_survival_by_class")

# chart 4: age distribution
# a histogram to understand the age range of passengers.

sns.histplot(
    data=df,
    x="Age",
    bins=30,
    kde=True,
    color="#3d5a80"
)

plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
save_chart("4_age_histogram")

# chart 5: survival rate by age group
# comparing children, teenagers, adults and older passengers.

sns.barplot(
    data=df,
    x="AgeGroup",
    y="Survived",
    hue="AgeGroup",
    palette="viridis",
    errorbar=None,
    legend=False
)

plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")
save_chart("5_survival_by_agegroup")

# chart 6: ticket fare by passenger class
# box plot helps us compare ticket prices in different classes.

sns.boxplot(
    data=df,
    x="Pclass",
    y="Fare",
    hue="Pclass",
    palette="pastel",
    legend=False
)

plt.ylim(0, 300)
plt.title("Ticket Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Ticket Fare")
save_chart("6_fare_by_class")

# chart 7: survival rate according to family size
# checking if travelling alone or with family affected survival.

family_survival = df.groupby("FamilySize")["Survived"].mean()

family_survival.plot(
    kind="line",
    marker="o",
    color="#e07a5f"
)

plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size (including passenger)")
plt.ylabel("Survival Rate")
save_chart("7_survival_by_family")

# chart 8: correlation heatmap
# this gives a quick idea about relationships between numerical columns.

corr_data = df[["Survived", "Pclass", "Age", "Fare", "FamilySize"]]
correlation = corr_data.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Correlation Heatmap")
save_chart("8_correlation_heatmap")

# -------------------- 5. PROJECT COMPLETED --------------------

print("\n-----------------------------------")
print("Titanic Survival Analysis Completed!")
print("All 8 charts have been saved in the 'charts' folder.")
print("-----------------------------------")
