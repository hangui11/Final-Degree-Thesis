import utils as ut
import pandas as pd
import numpy as np
from  similarity import compute_similarity

def calculateRatingsMean(matrix):
    ratingsMean = {}
    for k, v in matrix.items():
        ratingsMean[k] = sum(v.values())/len(v)
    return ratingsMean

    
def generate_m(movies_idx, ratings):
    # Complete the datastructure for rating matrix 
    # @TODO
    m = {}
    data = []
    for i in movies_idx:
        ratingsMovie = ratings.loc[(ratings['movieId'] == i)]
        data = ratingsMovie[['userId', 'rating']].values.tolist()
        rate = {}
        for j in data:
            rate[j[0]] = j[1]
        if len(rate) != 0:  m[i] = rate    
    return m 

def findItemsSeenAndNoSeenByUser(userId, matrix):
    seenMovies = {}
    unseenMovies = {}
    for item, users in matrix.items():
        if userId in users.keys(): seenMovies[item] = users
        else: unseenMovies[item] = users
    return (seenMovies, unseenMovies)

def item_based_recommender(target_user_idx, matrix):
    # target_user = matrix[target_user_idx]
    recommendations = []
    # Compute the similarity between  the target user and each other user in the matrix. 
    # We recomend to store the results in a dataframe (userId and Similarity)
    # @TODO 
    moviesRatingMean = calculateRatingsMean(matrix)
    seenMovies, unseenMovies = findItemsSeenAndNoSeenByUser(target_user_idx, matrix)
    predictRateUnseenMovies = {}
    
    userRate = {}
    for k, v in seenMovies.items():
        userRate[k] = matrix[k][target_user_idx]
    
    for kUnseenMovies, vUnseenMovies in unseenMovies.items():
        usersListA = list(vUnseenMovies.items())
        similarity = {}
        simMax = 0
        simMin = 0
        # Trobar les similaritats de les pel·lícules vistes amb la pel·lícula no vista
        for kSeenMovies, vSeenMovies in seenMovies.items():
            usersListB = list(vSeenMovies.items())
            sim = compute_similarity(usersListA, usersListB, moviesRatingMean[kUnseenMovies], moviesRatingMean[kSeenMovies])
            if simMax < sim: simMax = sim
            if simMin > sim: simMin = sim
            similarity[kSeenMovies] = sim
        
        # Normalitzar les similituds
        sumRateSim = 0
        similitude = 0
        for kSimilarity, vSimilarity in similarity.items():
            if vSimilarity != 0: 
                similarity[kSimilarity] = (vSimilarity - simMin) / (simMax-simMin)
                sumRateSim += similarity[kSimilarity]*userRate[kSimilarity]
                similitude += similarity[kSimilarity]
        
        # Predictir el rating de la pel·lícula no vista
        if sumRateSim == 0: predictRateUnseenMovies[kUnseenMovies] = 0
        else: predictRateUnseenMovies[kUnseenMovies] = sumRateSim/similitude
        recommendations.append((kUnseenMovies, predictRateUnseenMovies[kUnseenMovies]))

    recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)
    # Normalitzar les prediccions
    max = recommendations[0][1]
    min = recommendations[len(recommendations)-1][1]
    for i in range(len(recommendations)):
        predictRate = (recommendations[i][1] - min) / (max - min)
        recommendations[i] = (recommendations[i][0], predictRate)
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
    
    # item-to-item similarity
    target_user_idx = 1

    recommendations = item_based_recommender(target_user_idx, m)
    
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
