import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'dataset_scores.csv'
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
print(df.head())

# Plotting BLEURT Score distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Bleurt Score (-1,1)'], kde=True, bins=30)
plt.title('Distribution of BLEURT Scores')
plt.xlabel('BLEURT Score')
plt.ylabel('Frequency')
plt.show()

# Plotting ROUGE Scores distribution
plt.figure(figsize=(10, 5))

# ROUGE Score
sns.histplot(df['rougeL_f'], kde=True, bins=30)
plt.title('Distribution of ROUGE-L_F Scores')
plt.xlabel('ROUGE-L_F Score')
plt.ylabel('Frequency')

plt.show()

# Plotting ROUGE Scores distribution
plt.figure(figsize=(10, 5))

# ROUGE Score
sns.histplot(df['rouge1_f'], kde=True, bins=30)
plt.title('Distribution of ROUGE-1_F Scores')
plt.xlabel('ROUGE-1_F Score')
plt.ylabel('Frequency')

plt.show()

# Plotting ROUGE Scores distribution
plt.figure(figsize=(10, 5))

# ROUGE Score
sns.histplot(df['rouge2_f'], kde=True, bins=30)
plt.title('Distribution of ROUGE-2_F Scores')
plt.xlabel('ROUGE-2_F Score')
plt.ylabel('Frequency')

plt.show()

# Plotting ROUGE Scores distribution
plt.figure(figsize=(10, 5))

# ROUGE Score
sns.histplot(df['manual'], kde=True, bins=30)
plt.title('Distribution manual scores')
plt.xlabel('Manual Score')
plt.ylabel('Frequency')

plt.show()
