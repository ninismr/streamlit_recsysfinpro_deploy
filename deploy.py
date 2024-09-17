import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
df_all = pd.read_csv('df_olist_clean.csv')
df_content = pd.read_csv('df_content.csv')

# Load SVD++ model
with open('best_model_svdpp.pkl', 'rb') as model_file:
    svdpp_model = pickle.load(model_file)

# TF-IDF vectorizer for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_content['product_category_name'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)

# Helper Functions
def get_user_purchased_products(user_id, df_all):
    user_purchases = df_all[df_all['customer_unique_id'] == user_id]
    purchased_product_ids = user_purchases['product_id'].unique()
    return purchased_product_ids, user_purchases

def get_last_purchase(user_purchases):
    if not user_purchases.empty:
        last_purchase = user_purchases.iloc[-1]
        last_purchase_df = pd.DataFrame([{
            'Product ID': last_purchase['product_id'],
            'Product Category': last_purchase['product_category_name'],
            'Purchase Date': last_purchase['order_purchase_timestamp']
        }])
        return last_purchase_df
    return pd.DataFrame(columns=['Product ID', 'Product Category', 'Purchase Date'])

def get_recommendations(product_id, df_content, cosine_sim, top_n=5):
    if product_id not in df_content['product_id'].values:
        return pd.DataFrame()  # Return empty DataFrame if product not found

    idx = df_content[df_content['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx].toarray().flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Skip the item itself

    recommended_indices = [i[0] for i in sim_scores]
    recommended_products = df_content.iloc[recommended_indices][['product_id', 'product_category_name']]
    
    return recommended_products

def svdpp_recommend(user_id, df_content, svdpp_model, top_n=5):
    user_rated_items = df_all[df_all['customer_unique_id'] == user_id]['product_id'].unique()
    category_predictions = []

    for product_id in df_content['product_id'].unique():
        if product_id not in user_rated_items:
            pred = svdpp_model.predict(user_id, product_id)
            category = df_content[df_content['product_id'] == product_id]['product_category_name'].values[0]
            category_predictions.append((product_id, category, pred.est))

    sorted_predictions = sorted(category_predictions, key=lambda x: x[2], reverse=True)[:top_n]
    return sorted_predictions

def hybrid_recommendation_system(user_id, df_all, df_content, cosine_sim, svdpp_model, top_n=5):
    # Step 1: Content-Based Recommendation
    purchased_product_ids, user_purchases = get_user_purchased_products(user_id, df_all)
    content_based_recommendations = pd.DataFrame(columns=['product_id', 'product_category_name'])

    if purchased_product_ids:
        # Get recommendations based on the first purchased product
        product_id_str = purchased_product_ids[0]
        recommendations = get_recommendations(product_id_str, df_content, cosine_sim, top_n=top_n)
        content_based_recommendations = recommendations[['product_id', 'product_category_name']]

    # Step 2: Collaborative Filtering - Get top categories
    user_rated_items = df_all[df_all['customer_unique_id'] == user_id]['product_id'].values
    category_predictions = []

    for product_id in df_content['product_id'].unique():
        if product_id not in user_rated_items:
            pred = svdpp_model.predict(user_id, product_id)
            category = df_content[df_content['product_id'] == product_id]['product_category_name'].values[0]
            category_predictions.append((category, pred.est))

    # Sort and select top categories based on the prediction score
    top_categories = sorted(category_predictions, key=lambda x: x[1], reverse=True)
    top_categories = list(dict.fromkeys([category for category, _ in top_categories]))[:5]  # Get top 5 unique categories

    # Step 3: Cascade into Content-Based for top categories
    cascade_recommendations = []

    for category in top_categories:
        # Filter products by category
        category_products = df_content[df_content['product_category_name'] == category]
        if len(category_products) > 0:
            # Sample up to 5 unique products from the category
            category_products_sample = category_products.sample(min(5, len(category_products)))
            cascade_recommendations.extend(category_products_sample[['product_id', 'product_category_name']].values.tolist())

    # Ensure only 5 categories are present in the final recommendations
    unique_categories = set()
    final_recommendations = []

    for rec in cascade_recommendations:
        if rec[1] not in unique_categories:
            unique_categories.add(rec[1])
        if len(unique_categories) > 5:
            break
        final_recommendations.append(rec)

    return {
        "content_based_recommendations": content_based_recommendations,
        "collaborative_filtering_recommendations": final_recommendations
    }

# Cold start recommendations (popular items for new users)
def find_popular_items(df_all, top_n=5):
    popular_items = df_all['product_id'].value_counts().head(top_n).index
    popular_products = df_all[df_all['product_id'].isin(popular_items)][['product_id', 'product_category_name']].drop_duplicates()
    return popular_products

# Check user ID and recommend
def check_and_recommend(user_id, df_all, df_content, cosine_sim, svdpp_model, global_area):
    purchased_products, user_purchases = get_user_purchased_products(user_id, df_all)

    if len(purchased_products) == 0:
        # No purchase history - Cold start recommendation
        st.subheader("New User Detected: Cold Start Recommendations")

        hot_items = find_popular_items(df_all, 5)
        st.subheader("Hot Items You Might Like:")
        st.write(hot_items)

        popular_in_area = find_popular_items(df_all[df_all['customer_state'] == global_area], 5)
        st.subheader(f"Popular Items in {global_area}:")
        st.write(popular_in_area)
    else:
        # Existing user - Hybrid recommendations
        purchased_products, user_purchases = get_user_purchased_products(user_id, df_all)
        st.write(f"**User {user_id}'s Last Purchase**")
        last_purchase_df = get_last_purchase(user_purchases)
        st.dataframe(last_purchase_df)
        
        st.subheader("Recommendations:")
        recommendations = hybrid_recommendation_system(user_id, df_all, df_content, cosine_sim, svdpp_model, top_n=5)
        
        st.subheader("Similar Items:")
        if not recommendations["content_based_recommendations"].empty:
            st.write(recommendations["content_based_recommendations"])
        else:
            st.write("No content-based recommendations available.")

        st.subheader("You might also like:")
        if recommendations["collaborative_filtering_recommendations"]:
            rec_df = pd.DataFrame(recommendations["collaborative_filtering_recommendations"], columns=['product_id', 'product_category_name'])
            st.write(rec_df)
        else:
            st.write("No collaborative filtering recommendations available.")

def get_category_recommendations(user_id, df_all, df_content, cosine_sim, svdpp_model, top_n=5):
    purchased_products, user_purchases = get_user_purchased_products(user_id, df_all)
    purchased_categories = df_all[df_all['product_id'].isin(purchased_products)]['product_category_name'].unique()

    user_recommendations = {}
    if len(purchased_categories) > 0:
        first_category = purchased_categories[0]
        first_category_products = df_all[df_all['product_category_name'] == first_category]['product_id'].unique()
        if len(first_category_products) > 0:
            recommended_items = get_recommendations(first_category_products[0], df_content, cosine_sim, top_n)
            user_recommendations[first_category] = recommended_items

    all_products = df_all['product_id'].unique()
    unseen_products = [p for p in all_products if p not in purchased_products]
    predicted_ratings = [(prod, svdpp_model.predict(user_id, prod).est) for prod in unseen_products]
    top_rated_products = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]

    top_categories = df_all[df_all['product_id'].isin([p[0] for p in top_rated_products])]['product_category_name'].unique()[:4]

    for category in top_categories:
        category_products = df_all[df_all['product_category_name'] == category]['product_id'].unique()
        if len(category_products) > 0:
            recommended_items = get_recommendations(category_products[0], df_content, cosine_sim, top_n)
            user_recommendations[category] = recommended_items

    return user_recommendations, user_purchases

def batch_predict_category_recommendations(user_ids, df_all, df_content, cosine_sim, svdpp_model, top_n=5):
    recommendations = []

    for user_id in user_ids:
        user_category_recommendations, _ = get_category_recommendations(user_id, df_all, df_content, cosine_sim, svdpp_model, top_n)

        row = {'user_id': user_id}
        col_idx = 1
        for category, recommended_items in user_category_recommendations.items():
            row[f'category_{col_idx}'] = category
            row[f'products_{col_idx}'] = ', '.join(recommended_items['product_id'].tolist())
            col_idx += 1

        recommendations.append(row)

    df_recommendations = pd.DataFrame(recommendations)

    return df_recommendations

def show_detailed_recommendations(user_id, df_all, df_content, cosine_sim, svdpp_model):
    """Retrieve detailed recommendations for a selected user."""
    recommendations, user_purchases = get_category_recommendations(user_id, df_all, df_content, cosine_sim, svdpp_model)
    return recommendations, user_purchases


# Streamlit App
st.set_page_config(page_title="Olist Recommendation System", layout="wide")

# Sidebar for navigation
page_selection = st.sidebar.selectbox("Choose a page", ["Batch Prediction", "Single Prediction"])

# Batch Prediction Page
if page_selection == "Batch Prediction":
    st.title("Batch Prediction: Olist Recommendation System")

    # Initialize session state if not already done
    if 'batch_recommendations' not in st.session_state:
        st.session_state.batch_recommendations = None
    if 'selected_user_id' not in st.session_state:
        st.session_state.selected_user_id = None
    if 'user_recommendations' not in st.session_state:
        st.session_state.user_recommendations = None
    if 'user_purchases' not in st.session_state:
        st.session_state.user_purchases = None

    # Button to generate batch recommendations
    if st.button("Generate Batch Recommendations"):
        df_all_sample = df_all.sample(100, random_state=42)
        user_ids = df_all_sample['customer_unique_id'].unique()
        st.session_state.batch_recommendations = batch_predict_category_recommendations(user_ids, df_all, df_content, cosine_sim, svdpp_model)
        st.session_state.selected_user_id = None  # Reset selected user ID when new batch is generated

    # Display batch results table
    if st.session_state.batch_recommendations is not None:
        st.write("Click on a User ID to see detailed recommendations.")
        df_recommendations = st.session_state.batch_recommendations

        # Display the DataFrame with clickable User IDs
        user_ids = df_recommendations['user_id'].tolist()
        selected_user_id = st.selectbox("Select a User ID for Details", options=[None] + user_ids, index=0)

        if selected_user_id:
            st.session_state.selected_user_id = selected_user_id
            st.session_state.user_recommendations, st.session_state.user_purchases = show_detailed_recommendations(selected_user_id, df_all, df_content, cosine_sim, svdpp_model)

        st.dataframe(df_recommendations)

    # Display detailed recommendations for the selected User ID
    if st.session_state.selected_user_id:
        st.subheader(f"Detailed Recommendations for User ID: {st.session_state.selected_user_id}")

        # Show last purchase information
        if st.session_state.user_purchases is not None:
            last_purchase_df = get_last_purchase(st.session_state.user_purchases)
            if not last_purchase_df.empty:
                st.write("**Last Purchase Details**:")
                st.dataframe(last_purchase_df)

        # Show detailed recommendations in a table format
        if st.session_state.user_recommendations:
            recommendations = st.session_state.user_recommendations
            all_recommendations = []

            for category, items in recommendations.items():
                for _, row in items.iterrows():
                    all_recommendations.append({
                        'Category': category,
                        'Product ID': row['product_id']
                    })

            df_recommendations_details = pd.DataFrame(all_recommendations)
            st.dataframe(df_recommendations_details)

# Single Prediction Page
elif page_selection == "Single Prediction":
    st.title("Single Prediction: Olist Recommendation System")

    # Input user ID and state
    user_id = st.text_input("Enter User ID", "")
    customer_state = st.selectbox("Select Customer State", df_all['customer_state'].unique())

    # Button to get recommendations
    if st.button("Get Recommendations"):
        if user_id and customer_state:
            check_and_recommend(user_id, df_all, df_content, cosine_sim, svdpp_model, customer_state)
        else:
            st.write("Please provide a valid User ID and select a state.")