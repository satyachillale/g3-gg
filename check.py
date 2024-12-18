import pandas as pd

# Read CSV into a DataFrame
df = pd.read_csv('/Users/satyachillale/Downloads/datasets_new/remaining_dataset.csv')
all_jpg = df['IMG_ID'].apply(lambda x: isinstance(x, str) and x.endswith('.jpg')).all()
# print(df.head(5))
# Print the entire row where 'longitude' is -79.377834
rdf = df[df['LON'] == -79.377834]
print(rdf['neighbourhood'])
# print(df['longitude'].unique().shape)
# Print result
# if all_jpg:
#     print("All files in the 'IMG_ID' column end with .jpg.")
# else:
#     print("Not all files in the 'IMG_ID' column end with .jpg.")
# print(df.columns)
# # Specify the column name and the element to search
# column_name = 'IMG_ID'
# element = 'cc_32_4089361852.jpg'
# print(df.head())
# # Check if the element exists in the column
# if element in df[column_name].values:
#     print(f"'{element}' is present in column '{column_name}'.")
# else:
#     print(f"'{element}' is NOT present in column '{column_name}'.")
