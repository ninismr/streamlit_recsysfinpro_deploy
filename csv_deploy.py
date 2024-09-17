import streamlit as st
import pandas as pd

def load_data(file):
    """Load the CSV file into a DataFrame."""
    df = pd.read_csv(file)
    return df

def main():
    st.markdown('## Product Recommendations for Olist Users')

    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload the batch CSV file", type="csv")

    if uploaded_file is not None:
        # Load the data
        data = load_data(uploaded_file)

        # Create a dropdown to select user ID with a placeholder
        user_ids = data['user_id'].unique()
        user_ids = ['None'] + list(user_ids)  # Add placeholder option

        selected_user_id = st.selectbox('Select a User ID', user_ids)

        # Ensure the user selects a valid user ID
        if selected_user_id != 'None':
            # Filter data for the selected user ID
            user_data = data[data['user_id'] == selected_user_id]

            # Display recommendations in a tidy format
            st.write(f"**Product Recommendations for User ID: {selected_user_id}**")

            # Create an empty list to store recommendation data
            recommendation_data = []

            # Iterate over the rows of the selected user ID
            for _, row in user_data.iterrows():
                for i in range(1, 6):
                    category_col = f'category_{i}'
                    products_col = f'products_{i}'
                    
                    if pd.notna(row[category_col]) and pd.notna(row[products_col]):
                        category = row[category_col]
                        products = eval(row[products_col])  # Convert string representation of list to list
                        recommendation_data.append({
                            'Category': category,
                            'Products': ', '.join(products)
                        })

            # Convert the recommendation data into a DataFrame
            recommendation_df = pd.DataFrame(recommendation_data)

            # Display the recommendations in a table
            st.dataframe(recommendation_df)

if __name__ == "__main__":
    main()
