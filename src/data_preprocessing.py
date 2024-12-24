import pandas as pd

# Load both datasets
data = pd.read_csv('./data/fighter_stats.csv') 
data2 = pd.read_csv('./data/ufc-master.csv')

# Display information about both datasets
print("Dataset 1 Info:")
print(data.info())
print(data.head())

print("Dataset 2 Info:")
print(data2.info())
print(data2.head())

# Rename columns for consistency
data.rename(columns={
    'R_fighter': 'RedFighter',
    'B_fighter': 'BlueFighter',
    'date': 'Date'
}, inplace=True)

# Merge datasets on common columns
combined_data = pd.merge(data, data2, on=['RedFighter', 'BlueFighter', 'Date'], how='outer')

# Handle missing values
combined_data = combined_data.bfill().ffill()

# Save the combined dataset
combined_data.to_csv('./data/combined_fighter_stats.csv', index=False)
print(f"Combined dataset saved to {'./data/combined_fighter_stats.csv'}")
