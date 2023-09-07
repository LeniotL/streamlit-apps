
# Function to recommend restaurants based on selected genres and optionally, a nearest station
def recommend_by_genre(input_genres, nearest_station=None, cosine_sim=cosine_sim, min_similarity=0.5):
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
