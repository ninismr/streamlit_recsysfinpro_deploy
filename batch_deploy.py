import streamlit as st
import pandas as pd

df_olist_clean = pd.read_csv('df_olist_clean.csv')  

def load_data(file):
    """Load the CSV file into a DataFrame."""
    df = pd.read_csv(file)
    return df

def main():
    st.markdown('## Product Recommendations for Olist Users')

    uploaded_file = st.file_uploader("Upload the batch CSV file", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        user_ids = data['user_id'].unique()
        user_ids = ['None'] + list(user_ids)  

        selected_user_id = st.selectbox('Select a User ID', user_ids)

        if selected_user_id != 'None':
            user_order_data = df_olist_clean[df_olist_clean['customer_unique_id'] == selected_user_id][['customer_unique_id', 'order_id', 'product_id', 'product_category_name', 'order_purchase_timestamp']].drop_duplicates(subset='order_id')

            st.write(f"**Order Data for User ID: {selected_user_id}**")
            st.dataframe(user_order_data)

            user_data = data[data['user_id'] == selected_user_id]

            st.write(f"**Product Recommendations for User ID: {selected_user_id}**")

            recommendation_data = []

            for _, row in user_data.iterrows():
                for i in range(1, 6):
                    category_col = f'category_{i}'
                    products_col = f'products_{i}'
                    
                    if pd.notna(row[category_col]) and pd.notna(row[products_col]):
                        category = row[category_col]
                        products = eval(row[products_col]) 
                        recommendation_data.append({
                            'Category': category,
                            'Products': ', '.join(products)
                        })

            recommendation_df = pd.DataFrame(recommendation_data)

            st.dataframe(recommendation_df)

if __name__ == "__main__":
    main()
