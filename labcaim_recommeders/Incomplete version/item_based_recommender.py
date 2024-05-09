import utils as ut
import pandas as pd
import numpy as np
from  similarity import compute_similarity

def generate_m(movies_idx, ratings):
    # Complete the datastructure for rating 
    # A propose of datastructure is matrix M[movie][user] = rating 
    # @TODO
    m = {}

    return m 

def item_based_recommender(target_user_idx, matrix):
    recommendations = []

    # First, find the seen movies and unseen movies for an user
    # For each unseen movie, compute the similarity with seen movies for an user
    # Then, use the similarity to predict the rating for an unseen movie for target user
    # We recomend to store the results in a dataframe (itemId and Similarity) for each unseen movie
    # @TODO 
    






    # Normalize the prediction rating between 0 and 1
    # @ TODO
    
    
    
    
    return recommendations


if __name__ == "__main__":
    
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)
    
    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))

    m = generate_m(movies_idx, ratings_train)
    
    # item-to-item similarity for target user
    target_user_idx = 1

    recommendations = item_based_recommender(target_user_idx, m)
    
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
