
import streamlit as st
import pandas as pd
import pickle

# Load the model and data using pickle
with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

restaurants = pd.read_csv('restaurants.csv')

def recommend(restaurant_name, cosine_sim=cosine_sim):
    idx = restaurants[restaurants['name'] == restaurant_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get scores of the 10 most similar restaurants (excluding itself)
    restaurant_indices = [i[0] for i in sim_scores]
    return restaurants['name'].iloc[restaurant_indices]

# Streamlit App
    
# Title of the webpage
st.title("🍽️🥢🍚 Tokyo Restaurants Recommender")

selected_restaurant = st.selectbox(
    "Select a restaurant to get recommendations:",
    restaurants['name'].unique()
)

if st.button("Recommend"):
    recommended_restaurants = recommend(selected_restaurant)
    st.write(recommended_restaurants)
