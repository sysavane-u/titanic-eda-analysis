import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
# Data cleaning
df = pd.read_csv("train.csv")
df = df.drop(columns="Cabin")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Exploratory Data Analysis
print(f"The overall survival rate was {round(df['Survived'].mean() * 100, 2)}%.")
female_sr = round(df.groupby("Sex")["Survived"].mean()["female"] * 100, 2)
male_sr = round(df.groupby("Sex")["Survived"].mean()["male"] * 100, 2)
f_class = round(df.groupby("Pclass")["Survived"].mean()[1] * 100, 2)
s_class = round(df.groupby("Pclass")["Survived"].mean()[2] * 100, 2)
t_class = round(df.groupby("Pclass")["Survived"].mean()[3] * 100, 2)
df["Age_group"] = pd.cut(df["Age"],
                         bins=(0, 12, 17, 60, 100),
                         labels=("Children", "Teenagers", "Adults", "Elderly")
                         )
children = round(df.groupby("Age_group", observed=True)["Survived"].mean()["Children"] * 100, 2)
teen = round(df.groupby("Age_group", observed=True)["Survived"].mean()["Teenagers"] * 100, 2)
adult = round(df.groupby("Age_group", observed=True)["Survived"].mean()["Adults"] * 100, 2)
elder = round(df.groupby("Age_group", observed=True)["Survived"].mean()["Elderly"] * 100, 2)
s_embarked = round(df.groupby("Embarked")["Survived"].mean()["S"] * 100, 2)
c_embarked = round(df.groupby("Embarked")["Survived"].mean()["C"] * 100, 2)
q_embarked = round(df.groupby("Embarked")["Survived"].mean()["Q"] * 100, 2)
print(f"By gender;\n"
      f"{female_sr}% of Women survived while {male_sr}% of men survived.\n"
      f"By Class;\n"
      f"{f_class}% of survivors were first class, {s_class}% in second class, while {t_class}% in third class.\n"
      f"By embarking point;\n"
      f"{s_embarked}% of Survivors embarked in Southampton, {c_embarked}% in Cherbourg, while {q_embarked}% in "
      f"Queenstown.\n "
      f"By Age group;\n"
      f"{children}% of Survivors were Children, {teen}% were teenagers, {adult}% were adults, while {elder}% were "
      f"elderly.")


# Visualisation
def plot_survival_count():
    survival_count = df["Survived"].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(survival_count.index, survival_count.values, color=["red", "green"], width=0.4)
    plt.title("Overall Survivors/Deceased.", fontweight="bold")
    plt.xlabel("Survived/Deceased")
    plt.ylabel("Counts")
    for i, value in enumerate(survival_count.values):
        plt.text(i, value + 5, str(value), ha="center", fontsize=11, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xticks([0, 1], ["Deceased", "Survived"])
    plt.show()


def plot_gender_survival():
    gender_count = round(df.groupby("Sex")["Survived"].mean() * 100, 2)
    plt.figure(figsize=(8, 5))
    plt.bar(gender_count.index, gender_count.values, color=["pink", "blue"], width=0.4)
    plt.title("Bar chart showing survival rates of passengers according to gender", fontweight="bold", pad=20)
    plt.xlabel("Gender")
    plt.ylabel("Survival rates (%)")
    for i, value in enumerate(gender_count.values):
        plt.text(i, value + 5, str(f"{value}%"), ha="center", fontsize=11, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.show()


def plot_class_survival():
    class_count = round(df.groupby("Pclass")["Survived"].mean() * 100, 2)
    plt.figure(figsize=(8, 5))
    plt.bar(class_count.index, class_count.values, color=["gold", "silver", "brown"], width=0.4)
    plt.title("Bar chart showing survival rates of passengers according to boarding class", fontweight="bold", pad=20)
    plt.xticks([1, 2, 3], ["First class", "Second class", "Third class"])
    plt.xlabel("Boarding class", fontweight="bold")
    plt.ylabel("Survival rates (%)", fontweight="bold")
    for x, value in zip(class_count.index, class_count.values):
        plt.text(x, value + 1, str(f"{value}%"), ha="center", fontsize=11, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.show()


def plot_age_survival():
    survivors = df[df["Survived"] == 1]["Age"]
    non_survivors = df[df["Survived"] == 0]["Age"]

    plt.figure(figsize=(8, 5))
    plt.hist(non_survivors, bins=20, alpha=0.6, color="red", label="Did not survive", edgecolor="black")
    plt.hist(survivors, bins=20, alpha=0.6, color="darkgreen", label="Survived", edgecolor="black")
    plt.legend()
    plt.title("Age Distribution by Survival", fontweight="bold")
    plt.xlabel("Age", fontweight="bold")
    plt.ylabel("Count", fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.show()


def plot_corr_map():
    df_encoded = df.copy()
    df_encoded["Sex_encoded"] = df_encoded["Sex"].map({"male": 0, "female": 1})
    embarked_dummies = pd.get_dummies(df_encoded["Embarked"], prefix="Embarked")
    df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
    corr = df_encoded.drop(columns=["PassengerId", "SibSp", "Parch"]).corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Color map showing correlation between survival rate and other factors.", fontweight="bold", pad=20)
    plt.show()


def plot_fare_dist():
    plt.figure(figsize=(8, 5))
    first_c = df[df["Pclass"] == 1]["Fare"]
    second_c = df[df["Pclass"] == 2]["Fare"]
    third_c = df[df["Pclass"] == 3]["Fare"]
    plt.boxplot([first_c, second_c, third_c])
    plt.xticks([1, 2, 3], ["First Class", "Second Class", "Third Class"])
    plt.xlabel("Boarding Class", fontweight="bold")
    plt.title("Distribution of fares according to boarding class")
    plt.ylabel("Fares ($)", fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.show()


# Statistical Analysis
s_males = df[df["Sex"] == "male"]["Survived"]
f_males = df[df["Sex"] == "female"]["Survived"]

t_stat, p_value = stats.ttest_ind(s_males, f_males)
print(f"T test: {t_stat}")
print(f"P value: {p_value}")

if p_value < 0.05:
    print("Result: females significantly survives better than males✅")
else:
    print("Result: No significant difference — could be random chance ❌")

first_class = df[df["Pclass"] == 1]["Survived"]
third_class = df[df["Pclass"] == 3]["Survived"]

t_stat, p_value = stats.ttest_ind(first_class, third_class)
print(f"T test: {t_stat}")
print(f"P value: {p_value}")

if p_value < 0.05:
    print("Result: first class significantly survives better than third class✅")
else:
    print("Result: No significant difference — could be random chance ❌")

s_data = df["Survived"]
confidence = 0.95
mean = s_data.mean()
se = stats.sem(s_data)
interval = stats.t.interval(confidence,
                            df=len(s_data) - 1,
                            loc=mean,
                            scale=se)

print(f"Overall survival probability: {(round(mean, 2)) * 100}%")
print(f"95% Confidence Interval: {100 * (round(interval[0], 2))}% to {100 * (round(interval[1], 2))}%")

female = df[df["Sex"] == "female"]
s_female = female[female["Survived"] == 1]
prob_fs = round((len(s_female) / len(female)) * 100, 2)
print(f"Probability of survival if female: {prob_fs}%")

first_class = df[df["Pclass"] == 1]
first_class_s = first_class[first_class["Survived"] == 1]
prob_fs = round((len(first_class_s) / len(first_class)) * 100, 2)
print(f"Probability of survival if first class: {prob_fs}%")

male_t = df[(df["Sex"] == "male") & (df["Pclass"] == 3)]
mt_survivors = male_t[male_t["Survived"] == 1]
prob_mts = round((len(mt_survivors) / len(male_t)) * 100, 2)
print(f"Probability of survival if you're a man in third class: {prob_mts}%")

ps_female = 0.742
p_female = len(df[df["Sex"] == "female"]) / len(df)
ps = 0.3838

b_prob = round((ps_female * p_female / ps) * 100, 2)
print(f"Given a passenger survived, the probability that they were female is {b_prob}%")
plot_survival_count()
plot_gender_survival()
plot_class_survival()
plot_age_survival()
plot_corr_map()
plot_fare_dist()