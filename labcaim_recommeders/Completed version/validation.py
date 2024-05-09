import pandas as pd
import numpy as np
import numpy as np
import utils as ut
import trivial_recommender as trivial
import user_based_recommender as user
import item_based_recommender as item
import knn_hybrid_based_recommender as knn

## calculate the TfIdf of movies 
def toTfIdf(moviesGenre):
    moviesCount = moviesGenre.shape[0]
    genresTotal = moviesGenre.sum()
    maxGenre = genresTotal.max()
    toTfIdf = genresTotal/maxGenre * np.log2(moviesCount/genresTotal)
    toTfIdf = toTfIdf/((toTfIdf**2).sum())
    
    indexs = toTfIdf.index.tolist()
    values = toTfIdf.values.tolist()
    result = []
    for i in range(len(indexs)):
        if (np.isnan(values[i])): result.append((indexs[i], 0))
        else: result.append((indexs[i], values[i]))
    return result

## Calculate the cosinu similarity between two movies genres
def cosinuSimilarity(validationMoviesGenres, recommendsMoviesGenres):
    genres1 = toTfIdf(validationMoviesGenres)
    genres2 = toTfIdf(recommendsMoviesGenres)
    genres1 = sorted(genres1, key = lambda x:x[0])
    genres2 = sorted(genres2, key = lambda x:x[0])

    i, j, d1, d2, sum = 0, 0, 0, 0, 0
    while (i < len(genres1) and j < len(genres2)):
        if (genres1[i][0] == genres2[j][0]):
            d1 += genres1[i][1]*genres1[i][1]
            d2 += genres2[j][1]*genres2[j][1]
            sum += genres1[i][1]*genres2[j][1]
            i += 1
            j += 1
        elif (genres1[i][0] < genres2[j][0]):
            d1 += genres1[i][1]*genres1[i][1]
            i += 1
        else:
            d2 += genres2[j][1]*genres2[j][1]
            j += 1

    while (i < len(genres1)):
        d1 += genres1[i][1]*genres1[i][1]
        i += 1
    while (j < len(genres2)):
        d2 += genres2[j][1]*genres2[j][1]
        j += 1
        
    if (np.sqrt(d1)*np.sqrt(d2) == 0): return 0
    return sum/(np.sqrt(d1)*np.sqrt(d2))

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

    knn_rec = knn.mergeRecommendations(user_rec, item_rec)

    # Validation
    matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])
    validationMovies = ratings_val['movieId'].loc[ratings_val['userId'] == target_user_idx].values.tolist()
    validationMoviesGenres = matrixmpa_genres.loc[validationMovies]

    # Validation the recommender
    
    # Trivial recommender
    topTrivialMovies = list(list(zip(*trivial_rec[:5]))[0])
    recommendsMoviesTrivial = matrixmpa_genres.loc[topTrivialMovies]
    trivial_sim = cosinuSimilarity(recommendsMoviesTrivial, validationMoviesGenres)
    print("Trivial recommender similarity: ", trivial_sim)

    # User based recommender
    topUserMovies = list(list(zip(*user_rec[:5]))[0])
    recommendsMoviesUser = matrixmpa_genres.loc[topUserMovies]
    user_sim = cosinuSimilarity(validationMoviesGenres, recommendsMoviesUser)
    print("User based recommender similarity: ", user_sim)

    # Item based recommender
    topItemMovies = list(list(zip(*item_rec[:5]))[0])
    recommendsMoviesItem = matrixmpa_genres.loc[topItemMovies]
    item_sim = cosinuSimilarity(validationMoviesGenres, recommendsMoviesItem)
    print("Item based recommender similarity: ", item_sim)

    # KNN recommender
    topKnnMovies = list(list(zip(*knn_rec[:5]))[0])
    recommendsMoviesKnn = matrixmpa_genres.loc[topKnnMovies]
    knn_sim = cosinuSimilarity(validationMoviesGenres, recommendsMoviesKnn)
    print("KNN recommender similarity: ", knn_sim)



