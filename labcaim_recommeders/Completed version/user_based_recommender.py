import pandas as pd
import numpy as np
import utils as ut
from similarity import compute_similarity

# Calcute the ratings mean for each user in the ratings matrix
def calculateRatingsMean(matrix):
    ratingsMean = {}
    for k, v in matrix.items():
        ratingsMean[k] = sum(v.values())/len(v)
    return ratingsMean

# Generate a matrix with the ratings of each user for each movie 
def generate_m(users, ratings):
    # Complete the datastructure for rating matrix 
    m = {}
    userMovies = []
    for i in users:
        ratingsUser = ratings.loc[(ratings['userId'] == i)]
        userMovies = ratingsUser[['movieId', 'rating']].values.tolist()
        userMoviesRating = {}
        for j in userMovies:
            userMoviesRating[j[0]] = j[1]
        m[i] = userMoviesRating
    return m 

# Get all unseen movies for a user in the ratings matrix
def getUnseenmovies(seenMovies, matrix):
    unseenMovies = []
    first = True
    # Obtain unrated movies for an user using seen movies
    for id, auxUser in matrix.items():
        auxUserList = list(auxUser.items())
        auxUserRating = pd.DataFrame(auxUserList, columns=['movieId', 'rating'])
        unseenMovies1 = auxUserRating.loc[~auxUserRating['movieId'].isin(seenMovies)]
        unseenMovies1 = unseenMovies1['movieId'].values.tolist()
        if first: 
            unseenMovies = unseenMovies1
            first = not first
        else: 
            # Obtain no repeated unseen movies
            for i in unseenMovies1:
                if not i in unseenMovies: unseenMovies.append(i)
    
    return unseenMovies

# Compute the interest of unseen movies for a user in the ratings matrix using user-to-user similarity
def user_based_recommender(target_user_idx, matrix):
    target_user = matrix[target_user_idx]
    recommendations = []
    
    # Compute the similarity between  the target user and each other user in the matrix. 
    # We recomend to store the results in a dataframe (userId and Similarity)
    # @TODO 
    usersRatingsMean = calculateRatingsMean(matrix)
    similarity = {}
    simMax, simMin = 0, 0
    targetUserList = list(target_user.items())

    for userId, userMovies in matrix.items():
        if userId != target_user_idx:
            userMoviesList = list(userMovies.items())
            sim = compute_similarity(targetUserList, userMoviesList, usersRatingsMean[target_user_idx], usersRatingsMean[userId])
            if simMax < sim: simMax = sim
            if simMin > sim: simMin = sim
            similarity[userId] = sim
    for k,v in similarity.items():
        if v != 0: similarity[k] = (v - simMin) / (simMax-simMin)

    # Determine the unseen movies by the target user. Those films are identfied 
    # since don't have any rating. 
    # @TODO 
    targetUser= pd.DataFrame(targetUserList, columns=['movieId', 'rating'])
    seenMovies = targetUser[['movieId']]
    seenMovies = seenMovies['movieId'].values.tolist()
    unseenMovies = getUnseenmovies(seenMovies, matrix)

    # Generate recommendations for unrated movies based on user similarity and ratings.
    # @ TODO 
    meanUser = usersRatingsMean[target_user_idx]
    for i in unseenMovies:
        sum = 0
        for userId, userMovies in matrix.items():
            if userId != target_user_idx:
                sim = similarity[userId]
                ratingMean = usersRatingsMean[userId]
                if not i in userMovies:
                    ratingMovie = 0
                else:
                    ratingMovie = userMovies[i]
                sum += sim*(ratingMovie-ratingMean)
        recommendations.append((i, meanUser+sum))
    
    recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)

    # Normalize the prediction rating between 0 and 1
    max = recommendations[0][1]
    min = recommendations[len(recommendations)-1][1]
        
    for i in range(len(recommendations)):
        if (max - min != 0): interest = (recommendations[i][1] - min) / (max - min)
        else: interest = 1.0
        recommendations[i] = (recommendations[i][0], interest)
        
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
    m = generate_m(users_idy, ratings_train)
        
    # user-to-user similarity
    target_user_idx = 1
    recommendations = user_based_recommender(target_user_idx, m)
     
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    
     
    








