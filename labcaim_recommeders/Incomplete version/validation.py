import pandas as pd
import numpy as np
import numpy as np
import utils as ut
import trivial_recommender as trivial
import user_based_recommender as user
import item_based_recommender as item
import knn_hybrid_based_recommender as knn

def toTfIdf(moviesGenre):
    # We recommend to calculate the TfIdf for movies genres 
    # and it's possble complete the code using pandas library
    # @TODO
    tfIdf_result = []
    
    return tfIdf_result


def genres_similarity(validationMoviesGenres, recommendsMoviesGenres):
    # We recommend to calculate the TfIdf for movies genres to ease similarity computation
    genres1 = toTfIdf(validationMoviesGenres)
    genres2 = toTfIdf(recommendsMoviesGenres)
    genres1 = sorted(genres1, key = lambda x:x[0])
    genres2 = sorted(genres2, key = lambda x:x[0])

    # Complete the code using a similarity function to compute the similarity between two set of movies genres
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.
    # @TODO
    similarity = 1



    return similarity

if __name__ == "__main__":
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)
    
    # Movies data and users data
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))

    target_user_idx = 1

    # Test the recommender
    trivial_rec = trivial.trivial_recommender(ratings_train, dataset["movies.csv"])

    user_m = user.generate_m(users_idy, ratings_train)
    user_rec = user.user_based_recommender(target_user_idx, user_m)
    
    item_m = item.generate_m(movies_idx, ratings_train)
    item_rec = item.item_based_recommender(target_user_idx, item_m)

    knn_rec = knn.knn_hybrid_based_recommender(user_rec, item_rec)

    # Validation
    matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])
    validationMovies = ratings_val['movieId'].loc[ratings_val['userId'] == target_user_idx].values.tolist()
    validationMoviesGenres = matrixmpa_genres.loc[validationMovies]

    # Validation the recommenders
    # Trivial recommender
    topTrivialMovies = list(list(zip(*trivial_rec[:5]))[0])
    recommendsMoviesTrivial = matrixmpa_genres.loc[topTrivialMovies]
    trivial_sim = genres_similarity(recommendsMoviesTrivial, validationMoviesGenres)
    print("Trivial recommender similarity: ", trivial_sim)

    # User based recommender
    topUserMovies = list(list(zip(*user_rec[:5]))[0])
    recommendsMoviesUser = matrixmpa_genres.loc[topUserMovies]
    user_sim = genres_similarity(validationMoviesGenres, recommendsMoviesUser)
    print("User based recommender similarity: ", user_sim)

    # Item based recommender
    topItemMovies = list(list(zip(*item_rec[:5]))[0])
    recommendsMoviesItem = matrixmpa_genres.loc[topItemMovies]
    item_sim = genres_similarity(validationMoviesGenres, recommendsMoviesItem)
    print("Item based recommender similarity: ", item_sim)

    # KNN recommender
    topKnnMovies = list(list(zip(*knn_rec[:5]))[0])
    recommendsMoviesKnn = matrixmpa_genres.loc[topKnnMovies]
    knn_sim = genres_similarity(validationMoviesGenres, recommendsMoviesKnn)
    print("KNN recommender similarity: ", knn_sim)



