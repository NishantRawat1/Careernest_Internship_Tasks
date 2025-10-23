import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# --- Load Dataset ---
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# --- View Basic Info ---
print("First 5 rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nSummary Statistics:\n", df.describe())

# --- Check Missing Values ---
print("\nMissing values:\n", df.isnull().sum())

# --- Visualizations ---
sns.pairplot(df, hue='species', corner=True)
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

sns.boxplot(x='species', y='sepal length (cm)', data=df)
plt.title("Sepal Length by Species")
plt.show()

# --- Insights ---
print("\nInsights:")
print("• The dataset has 3 flower species.")
print("• Setosa is distinctly separable by petal length and width.")
print("• Versicolor and Virginica overlap slightly in features.")
