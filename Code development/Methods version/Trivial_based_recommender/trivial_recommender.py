import os
import sys
import pandas as pd 
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(code_development_dir)

from utils import *

def trivial_recommender(ratings: object, movies:object, k: int = 5) -> list: 
    # Provide the code for the trivial recommender here. This function should return 
    # the list of the top most viewed films according to the ranking (sorted in descending order).
    # Consider using the utility functions from the pandas library.
    ratingsMean = ratings[['movieId', 'rating']].groupby(by=['movieId']).mean()
    ratingsMovies = pd.merge(ratingsMean, movies, on='movieId', how='inner')
    sortValues = ratingsMovies.sort_values(by=['rating'], ascending=False)
    relevantsK = sortValues.head(k)
    return relevantsK



if __name__ == "__main__":
    
     # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = split_users(dataset["ratings.csv"], val_movies)
    
    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    # 93
    target_user_idx = 430
    print('The prediction for user ' + str(target_user_idx) + ':')

    start = time.time()

     ## trivial recommender
    topMovieTrivial = trivial_recommender(ratings_train, dataset["movies.csv"], 5)
    topMovieTrivial = topMovieTrivial['movieId'].values.tolist()
    for recomendation in topMovieTrivial:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
    
     # Validation
    matrixmpa_genres, validationMoviesGenres = validationMoviesGenres(dataset["movies.csv"], ratings_val, target_user_idx)
    
    recommendsMoviesTrivial = matrixmpa_genres.loc[topMovieTrivial]
    
    # sim entre matriu genere amb recomanador sistema
    sim = cosinuSimilarity(validationMoviesGenres, recommendsMoviesTrivial)
    print(' Similarity with trivial recommender: ' + str(sim))
    
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")

