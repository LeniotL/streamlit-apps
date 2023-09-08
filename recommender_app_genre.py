
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import io
import s3fs
import h5py
import requests
import json
from geocoding import get_coordinates_from_address


# Load pre-computed cosine similarities and the restaurant dataset
with h5py.File('compressed_cosine_sim.h5', 'r') as file:
    cosine_sim = file['cosine_similarity'][:]

restaurants = pd.read_csv('restaurants_model.csv')


# Drop unnecessary columns to leave only genre features
genres_df = restaurants.drop(columns=['name', 'genre', 'rating_val', 'nearest_station', 'address'])


# Load the Lottie JSON data from the local file
with open('lottie_data.json', 'r') as file:
    lottie_data_from_file = json.load(file)

    
# Display the Lottie animation using the loaded data
st_lottie(lottie_data_from_file, speed=1, width=800, height=500, key="initial")


# Function to recommend restaurants based on selected genres and optionally, a nearest station
def recommend_by_genre(input_genres, nearest_station=None, cosine_sim=cosine_sim, min_similarity=0):
    # Check if all selected genres exist in the dataset
    for genre in input_genres:
        if genre not in genres_df.columns:
            return f"Genre {genre} not found in the dataset"

    # Create a combined genre vector for all selected genres
    genre_vector = np.zeros(genres_df.shape[1])
    for genre in input_genres:
        genre_index = list(genres_df.columns).index(genre)
        genre_vector[genre_index] = 1

    # Normalize the genre vector and reshape
    genre_vector /= len(input_genres)
    genre_vector = genre_vector.reshape(1, -1)

    # Compute the cosine similarity scores for the genre vector against all restaurants
    sim_scores = cosine_similarity(genre_vector, genres_df.values)
    sim_scores = [x for x in enumerate(sim_scores[0]) if x[1] >= min_similarity]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # If a station is selected, filter by that station
    if nearest_station:
        station_filtered_indices = [i[0] for i in sim_scores if restaurants.iloc[i[0]]['nearest_station'] == nearest_station]
        top_10_indices = station_filtered_indices[:10]
    else:
        top_10_indices = [i[0] for i in sim_scores[:10]]

    # Get the top 10 restaurants and sort them by rating
    top_10_restaurants = restaurants.iloc[top_10_indices]
    sorted_restaurants = top_10_restaurants.sort_values(by='rating_val', ascending=False)
    return sorted_restaurants[['name', 'rating_val', 'nearest_station']]
    

# Streamlit UI: Select genres and a station
selected_genres = st.multiselect(
    "Select genres to get restaurant recommendations:", genres_df.columns
)
selected_station = st.selectbox("Choose a nearby station:", restaurants['nearest_station'].unique())


# Button to get recommendations
if st.button("Recommend"):
    recommended_data = recommend_by_genre(selected_genres, selected_station)
    recommended_data = recommended_data.rename(columns={"name": "Name", "rating_val": "Rating", "nearest_station": "Nearest Station"})
    
    # Ensure that 'address' column is present in recommended_data
    recommended_data = pd.merge(recommended_data, restaurants[['name', 'nearest_station', 'address']], 
                            left_on=['Name', 'Nearest Station'], 
                            right_on=['name', 'nearest_station'], 
                            how='left').drop_duplicates()

    # Geocode the addresses of the recommended restaurants only
    recommended_data['latitude'], recommended_data['longitude'] = zip(*recommended_data['address'].map(get_coordinates_from_address))

    st.table(recommended_data[['Name', 'Rating', 'Nearest Station']])

    # Display map
    st.map(recommended_data[['latitude', 'longitude']])
