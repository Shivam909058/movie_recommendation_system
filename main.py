import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('movies.csv')

# Feature selection
features = ['genres', 'keywords', 'cast', 'director', 'tagline']

# Replace null values with empty string
for feature in features:
    data[feature] = data[feature].fillna('')

# Combine features into a single column
def combine_features(row):
    return row['genres'] + " " + row['keywords'] + " " + row['cast'] + " " + row['director'] + " " + row['tagline']

data['combined_features'] = data.apply(combine_features, axis=1)

# Convert the data into numerical data
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Finding the similarity between movies using cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# Create Streamlit app
st.title('Movie Recommendation System')

# User input for movie name
movie_name = st.text_input('Enter the movie name:')

if movie_name:
    movie_list = data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, movie_list)
    if close_matches:
        matched_movie = close_matches[0]
        index = data[data['title'] == matched_movie].index[0]
        similarity_scores = list(enumerate(similarity[index]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]  # Exclude the movie itself
        st.write(f'Top 10 similar movies to {movie_name}:')
        for i, movie in enumerate(sorted_movies, 1):
            st.write(f"{i}. {data.loc[movie[0], 'title']}")
    else:
        st.write("No close match found.")
