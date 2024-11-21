import pandas as pd
import streamlit as st

# Load the CSV file
file_path = 'train.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Group by category and sub_category to count the occurrences
category_counts = df.groupby(['category', 'sub_category']).size().reset_index(name='count')

# Sort values by category and sub_category for better readability
category_counts = category_counts.sort_values(by=['category', 'sub_category']).reset_index(drop=True)

# Display in Streamlit with a well-tabulated format
st.title("Category and Sub-Category Counts")

# Create a pivot table for better readability, which separates each category and sub-category
pivot_table = category_counts.pivot_table(index='category', columns='sub_category', values='count', fill_value=0)

# Display the pivot table using Streamlit's `st.dataframe` for an interactive table, or `st.table` for a static table
st.write("### Interactive Table of Counts by Category and Sub-Category")
st.dataframe(category_counts, use_container_width=True)

# Alternatively, display a more aggregated format if needed
st.write("### Pivot Table View of Counts by Category and Sub-Category")
st.dataframe(pivot_table, use_container_width=True)
