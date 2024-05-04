import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
from utils import *
import pandas as pd
import numpy as np
import time

class UserToUser:
    def __init__(self, ratings_train, movies, users, k=5) -> None:
        self.ratings_train = ratings_train
        self.topK = k
        self.movies = movies
        self.users = users
        self.matrix = self.generate_users_matrix()

    def calculateRatingsMean(self):
        ratingsMean = {}
        matrix = self.matrix
        for k, v in matrix.items():
            ratingsMean[k] = sum(v.values())/len(v)
        return ratingsMean

    def generate_users_matrix(self):
        # Complete the datastructure for rating matrix 
        m = {}
        data = []
        ratings = self.ratings_train
        users = self.users
        for i in users:
            ratingsUser = ratings.loc[(ratings['userId'] == i)]
            data = ratingsUser[['movieId', 'rating']].values.tolist()
            rate = {}
            for j in data:
                rate[j[0]] = j[1]
            m[i] = rate
        return m 
    
    def pearsonSimilarity(self, userA, userB, meanUserA, meanUserB):
        ## using dict
        ratingsA = {itemId: rating-meanUserA for itemId, rating in userA}
        ratingsB = {itemId: rating-meanUserB for itemId, rating in userB}
        
        # Find common users and their ratings
        common_items = set(ratingsA.keys()) & set(ratingsB.keys())
        if not common_items:
            return 0  # No common users, similarity is 0
        
        # Calculate person similarity
        sumAB, sumA, sumB = 0, 0, 0
        for itemId in common_items:
            ratingA = ratingsA[itemId]
            ratingB = ratingsB[itemId]
            sumAB += ratingA * ratingB
            sumA += ratingA ** 2
            sumB += ratingB ** 2
        
        # Check for division by zero
        if sumA == 0 or sumB == 0: return 0

        similarity = sumAB / (np.sqrt(sumA) * np.sqrt(sumB))
        return similarity
        
    def getUnseenmovies(self, seenMovies):
        matrix = self.matrix
        unseenMovies = []
        first = True
        
        # obtenir les pelicules no avaluades per target user
        for id, auxUser in matrix.items():
            auxUserList = list(auxUser.items())
            auxUserRating = pd.DataFrame(auxUserList, columns=['movieId', 'rating'])
            unseenMovies1 = auxUserRating.loc[~auxUserRating['movieId'].isin(seenMovies)]
            unseenMovies1 = unseenMovies1['movieId'].values.tolist()
            if first: 
                unseenMovies = unseenMovies1
                first = not first
            else: 
                # obtenir unseen movies no repetits
                for i in unseenMovies1:
                    if not i in unseenMovies: unseenMovies.append(i)

        return unseenMovies
    
    def user_based_recommender(self, target_user_idx):
        matrix = self.matrix
        target_user = matrix[target_user_idx]
        recommendations = []
        # Compute the similarity between  the target user and each other user in the matrix. 
        # We recomend to store the results in a dataframe (userId and Similarity)

        # Calcular la mitjana d'avaluació dels usuaris
        usersRatingsMean = self.calculateRatingsMean()

        # Calcular la similaritat de target user amb la resta d'usuaris
        similarity = {}
        simMax, simMin = 0, 0
        targetUserList = list(target_user.items())
        
        for userId, userMovies in matrix.items():
            if userId != target_user_idx:
                userMoviesList = list(userMovies.items())
                sim = self.pearsonSimilarity(targetUserList, userMoviesList, usersRatingsMean[target_user_idx], usersRatingsMean[userId])
                if simMax < sim: simMax = sim
                if simMin > sim: simMin = sim
                similarity[userId] = sim
        
        # Normalitzar les similaritats entre usuaris
        for k,v in similarity.items():
            if v != 0: similarity[k] = (v - simMin) / (simMax-simMin)
        
        # Determine the unseen movies by the target user. Those films are identfied since don't have any rating. 
        
        # Obtenir les pelicules avaluades per target user
        targetUser= pd.DataFrame(targetUserList, columns=['movieId', 'rating'])
        seenMovies = targetUser[['movieId']]
        seenMovies = seenMovies['movieId'].values.tolist()

        unseenMovies = self.getUnseenmovies(seenMovies)
        # Generate recommendations for unrated movies based on user similarity and ratings.
                    
        # Per cada pelicula no avaluada computa interes sobre ella (predicció)
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
        
        # Normalitzar les prediccions
        max = recommendations[0][1]
        min = recommendations[len(recommendations)-1][1]
        
        for i in range(len(recommendations)):
            if (max - min != 0): interest = (recommendations[i][1] - min) / (max - min)
            else: interest = 1.0
            recommendations[i] = (recommendations[i][0], interest)

        self.recommendations = recommendations
        return recommendations 
    
    def printTopRecommendations(self):
        for recomendation in self.recommendations[:self.topK]:
            rec_movie = self.movies[self.movies["movieId"]  == recomendation[0]]
            print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    def validation(self, ratings_val, target_user_idx):
        # Validation
        matrixmpa_genres, validationMoviesGenress = validationMoviesGenres(self.movies, ratings_val, target_user_idx)

        topMoviesUser = list(list(zip(*self.recommendations[:self.topK]))[0])
        recommendsMoviesUser = matrixmpa_genres.loc[topMoviesUser]
        
        # sim entre matriu genere amb recomanador user
        sim = cosinuSimilarity(validationMoviesGenress, recommendsMoviesUser)
        # print(' Similarity with user-to-user recommender: ' + str(sim))
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

    # 387, 109
    target_user_idx = 392
    print('The prediction for user ' + str(target_user_idx) + ':')

    userToUser = UserToUser(ratings_train, dataset['movies.csv'], users_idy)
    recommendations = userToUser.user_based_recommender(target_user_idx)
    userToUser.printTopRecommendations()
    userToUser.validation(ratings_val, target_user_idx)

    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")
