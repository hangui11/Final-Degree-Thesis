import pandas as pd 
import time 
import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
from utils import *

def naive_recommender(ratings: object, movies:object, k: int = 5) -> list: 
    # Provide the code for the naive recommender here. This function should return 
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

     ## naive recommender
    topMovieNaive = naive_recommender(ratings_train, dataset["movies.csv"], 5)
    topMovieNaive = topMovieNaive['movieId'].values.tolist()
    for recomendation in topMovieNaive:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
    
     # Validation
    matrixmpa_genres, validationMoviesGenres = validationMoviesGenres(dataset["movies.csv"], ratings_val, target_user_idx)
    
    recommendsMoviesNaive = matrixmpa_genres.loc[topMovieNaive]
    
    # sim entre matriu genere amb recomanador sistema
    sim = cosinuSimilarity(validationMoviesGenres, recommendsMoviesNaive)
    print(' Similarity with naive recommender: ' + str(sim))
    
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")

