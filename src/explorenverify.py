import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load combined dataset
combined_data = pd.read_csv('./data/combined_fighter_stats.csv')

# Example: Visualize fighter ages
sns.histplot(combined_data['R_age'].dropna(), kde=True, color='blue', label='Red Corner')
sns.histplot(combined_data['B_age'].dropna(), kde=True, color='red', label='Blue Corner')
plt.legend()
plt.title("Age Distribution of Fighters")
plt.show()

# Inspect key stats
print(combined_data.info())
print(combined_data.describe())

# Check for any remaining missing values
print(combined_data.isnull().sum())

# Display basic information
print("Dataset Info:")
print(combined_data.info())

# Show summary statistics
print("\nSummary Statistics:")
print(combined_data.describe())

# Check for missing values
print("\nMissing Values:")
print(combined_data.isnull().sum())

# Display the first few rows
print("\nFirst Few Rows:")
print(combined_data.head())

print(combined_data.isnull().sum())  # Confirm no null values
print(combined_data[['RedOdds', 'BlueOdds', 'RSubOdds', 'BSubOdds']].head())
