"""
PRODIGY_DS_02 - Data Science Internship @ Prodigy InfoTech
Task 02: Data cleaning + Exploratory Data Analysis (EDA) on the Titanic dataset.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ======================= 1. LOAD =======================
df = pd.read_csv("train.csv")
print("Shape:", df.shape)
print("\nMissing values before cleaning:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# ======================= 2. CLEANING =======================
# Age: fill with the median age of each Title group (Mr, Mrs, Miss, ...)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["Title"] = df["Title"].replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "the Countess": "Rare",
     "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare",
     "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"})
df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))

# Embarked: only 2 missing -> fill with the mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin: ~77% missing -> keep only the deck letter, unknown = 'U'
df["Deck"] = df["Cabin"].str[0].fillna("U")
df = df.drop(columns=["Cabin", "Ticket"])

# ======================= 3. FEATURE ENGINEERING =======================
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
df["Survived_Label"] = df["Survived"].map({0: "Died", 1: "Survived"})

print("\nMissing values after cleaning:\n", df.isnull().sum().sum())
df.to_csv("titanic_cleaned.csv", index=False)

# ======================= 4. EDA =======================
print("\nOverall survival rate: {:.1%}".format(df["Survived"].mean()))
print("\nSurvival by Sex:\n", df.groupby("Sex")["Survived"].mean().round(3))
print("\nSurvival by Pclass:\n", df.groupby("Pclass")["Survived"].mean().round(3))
print("\nSurvival by AgeGroup:\n", df.groupby("AgeGroup", observed=True)["Survived"].mean().round(3))
print("\nSurvival by Embarked:\n", df.groupby("Embarked")["Survived"].mean().round(3))
print("\nSurvival by IsAlone:\n", df.groupby("IsAlone")["Survived"].mean().round(3))

# ---- Fig 1: survival counts by sex & class ----
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
sns.countplot(x="Survived_Label", data=df, ax=ax[0], palette=["#C44E52", "#55A868"], hue="Survived_Label", legend=False)
ax[0].set_title("Overall Survival")
sns.barplot(x="Sex", y="Survived", data=df, ax=ax[1], palette="Set2", hue="Sex", legend=False)
ax[1].set_title("Survival Rate by Sex")
sns.barplot(x="Pclass", y="Survived", data=df, ax=ax[2], palette="Blues_d", hue="Pclass", legend=False)
ax[2].set_title("Survival Rate by Passenger Class")
plt.tight_layout(); plt.savefig("01_survival_overview.png", dpi=150); plt.close()

# ---- Fig 2: Age distribution ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["Age"], bins=30, kde=True, ax=ax[0], color="#4C72B0")
ax[0].set_title("Age Distribution")
sns.kdeplot(data=df, x="Age", hue="Survived_Label", fill=True, ax=ax[1], common_norm=False)
ax[1].set_title("Age Distribution by Survival")
plt.tight_layout(); plt.savefig("02_age_distribution.png", dpi=150); plt.close()

# ---- Fig 3: Sex x Class heatmap ----
pivot = df.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")
plt.figure(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", cbar_kws={"label": "Survival rate"})
plt.title("Survival Rate: Sex x Passenger Class")
plt.tight_layout(); plt.savefig("03_sex_class_heatmap.png", dpi=150); plt.close()

# ---- Fig 4: Fare, family size, embarkation ----
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
sns.boxplot(x="Pclass", y="Fare", data=df, ax=ax[0], hue="Pclass", palette="pastel", legend=False)
ax[0].set_yscale("log"); ax[0].set_title("Fare by Class (log scale)")
sns.barplot(x="FamilySize", y="Survived", data=df, ax=ax[1], color="#8172B2")
ax[1].set_title("Survival Rate by Family Size")
sns.barplot(x="Embarked", y="Survived", data=df, ax=ax[2], hue="Embarked", palette="muted", legend=False)
ax[2].set_title("Survival Rate by Port of Embarkation")
plt.tight_layout(); plt.savefig("04_fare_family_embarked.png", dpi=150); plt.close()

# ---- Fig 5: Age group survival ----
plt.figure(figsize=(8, 5))
sns.barplot(x="AgeGroup", y="Survived", data=df, hue="AgeGroup", palette="viridis", legend=False)
plt.title("Survival Rate by Age Group")
plt.tight_layout(); plt.savefig("05_agegroup_survival.png", dpi=150); plt.close()

# ---- Fig 6: correlation heatmap ----
num = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]].copy()
num["Sex_male"] = (df["Sex"] == "male").astype(int)
plt.figure(figsize=(9, 7))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout(); plt.savefig("06_correlation_heatmap.png", dpi=150); plt.close()

print("\nEDA complete. Cleaned data + 6 figures saved.")
