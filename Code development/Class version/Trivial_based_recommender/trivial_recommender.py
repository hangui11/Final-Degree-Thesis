import sys
import os
import pandas as pd 
import time 

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(code_development_dir)

from utils import *

class Trivial:
    def __init__(self, ratings_train, movies, k=5) -> None:
        self.ratings_train = ratings_train
        self.movies = movies
        self.topK = k
        self.recommendations = self.trivial_recommender()

    def trivial_recommender(self) -> list: 
        # Provide the code for the trivial recommender here. This function should return 
        # the list of the top most viewed films according to the ranking (sorted in descending order).
        # Consider using the utility functions from the pandas library.
        ratingsMean = self.ratings_train[['movieId', 'rating']].groupby(by=['movieId']).mean()
        ratingsMovies = pd.merge(ratingsMean, self.movies, on='movieId', how='inner')
        sortValues = ratingsMovies.sort_values(by=['rating'], ascending=False)
        relevantsK = sortValues.head(self.topK)
        self.topMovieTrivial = relevantsK['movieId'].values.tolist()
        return self.topMovieTrivial
    
    def printTopRecommendations(self):
        for recomendation in self.topMovieTrivial:
            rec_movie = self.movies[self.movies["movieId"]  == recomendation]
            print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    def validation(self, ratings_val, target_user_idx):
        matrixmpa_genres, validationMoviesGenress = validationMoviesGenres(self.movies, ratings_val, target_user_idx)
    
        recommendsMoviesTrivial = matrixmpa_genres.loc[self.topMovieTrivial]
    
        # sim entre matriu genere amb recomanador sistema
        sim = cosinuSimilarity(validationMoviesGenress, recommendsMoviesTrivial)
        # print(' Similarity with trivial recommender: ' + str(sim))
        return sim

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

    start = time.time()
    # 93
    target_user_idx = 401
    print('The prediction for user ' + str(target_user_idx) + ':')
    trivial_recommender = Trivial(ratings_train, dataset["movies.csv"])
    trivial_recommender.printTopRecommendations()
    trivial_recommender.validation(ratings_val, target_user_idx)
    
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")
