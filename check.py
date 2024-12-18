import pandas as pd

# Read CSV into a DataFrame
df = pd.read_csv('/Users/satyachillale/Downloads/datasets_new/remaining_dataset.csv')
print(df.columns)
# Specify the column name and the element to search
column_name = 'IMG_ID'
element = 'cc_32_4089361852.jpg'
print(df.head())
# Check if the element exists in the column
if element in df[column_name].values:
    print(f"'{element}' is present in column '{column_name}'.")
else:
    print(f"'{element}' is NOT present in column '{column_name}'.")
