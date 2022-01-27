import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("results.csv")
sns.lineplot(data=df, x="embedding_dim", y="percent")
plt.title("Proportion of test embeddings in the interpolation regime")
plt.show()
