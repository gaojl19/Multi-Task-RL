import pandas as pd
import seaborn as sns

df = pd.read_csv("./success_rate.csv", "r")
ax = sns.barplot(x="task", y="success_rate", hue="sex", data=df)