import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Correctly load the dataset with semicolon delimiter
file_path = 'Book27.csv'
df = pd.read_csv(file_path, delimiter=';')

# Now, let's ensure the DataFrame is loaded correctly with multiple columns
print(df.head())
print(df.shape)

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.show()

# Select a subset of variables for the pair plot
variables_to_plot = ['AGE', 'HEALTH', 'HEIGHT', 'WEIGHT', 'ANXIETYEV', 'HOUYRSLIV_Normalized']
sns.pairplot(df[variables_to_plot], diag_kind='kde')
plt.suptitle('Pair Plot of Selected Variables', y=1.02)  # Adjust the subtitle to not overlap with plots
plt.show()

# Box Plot for Health vs. Anxiety Ever
plt.figure(figsize=(10, 6))
sns.boxplot(x='HEALTH', y='ANXIETYEV', data=df)
plt.title('Health vs. Anxiety Ever')
plt.show()

# Violin Plot for Anxiety Ever vs. Weight Distribution
plt.figure(figsize=(10, 6))
sns.violinplot(x='ANXIETYEV', y='WEIGHT', data=df)
plt.title('Anxiety Ever vs. Weight Distribution')
plt.show()

# Bar Chart for Average Height by Anxiety Ever Status
df.groupby('ANXIETYEV')['HEIGHT'].mean().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Average Height by Anxiety Ever Status')
plt.ylabel('Average Height')
plt.xlabel('Anxiety Ever')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.show()
