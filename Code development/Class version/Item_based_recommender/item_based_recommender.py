import sys
import os
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(code_development_dir)

from utils import *

class ItemToItem():
    def __init__(self, ratings_train, movies, users, k=5) -> None:
        self.ratings_train = ratings_train
        self.topK = k
        self.movies = movies
        self.users = users
        self.matrix = self.generate_items_matrix()
        
    def calculateRatingsMean(self):
        matrix = self.matrix
        ratingsMean = {}
        for k, v in matrix.items():
            ratingsMean[k] = sum(v.values())/len(v)
        return ratingsMean

    def pearsonSimilarity(self, itemA, itemB, meanItemA, meanItemB):
        ratingsA = {userId: rating-meanItemA for userId, rating in itemA}
        ratingsB = {userId: rating-meanItemB for userId, rating in itemB}
        
        # Find common users and their ratings
        common_users = set(ratingsA.keys()) & set(ratingsB.keys())
        if not common_users:
            return 0  # No common users, similarity is 0
        
        # Calculate person similarity
        # sumAB, sumA, sumB = 0, 0, 0
        # for userId in common_users:
        #     ratingA = ratingsA[userId]
        #     ratingB = ratingsB[userId]
        #     sumAB += ratingA * ratingB
        #     sumA += ratingA ** 2
        #     sumB += ratingB ** 2
        
        sumAB = sum([ratingsA[userId] * ratingsB[userId] for userId in common_users])
        sumA = sum([ratingsA[userId] ** 2 for userId in common_users])
        sumB = sum([ratingsB[userId] ** 2 for userId in common_users])
        
        # Check for division by zero
        if sumA == 0 or sumB == 0: return 0

        similarity = sumAB / (np.sqrt(sumA) * np.sqrt(sumB))
        return similarity

    
    def generate_items_matrix(self):
        # Complete the datastructure for rating matrix 
        movies_idx = self.movies['movieId']
        m = {}
        data = []
        ratings = self.ratings_train
        for i in movies_idx:
            ratingsMovie = ratings.loc[(ratings['movieId'] == i)]
            data = ratingsMovie[['userId', 'rating']].values.tolist()
            rate = {}
            for j in data:
                rate[j[0]] = j[1]
            if len(rate) != 0:  m[i] = rate    
        return m 

    def findItemsSeenAndNoSeenByUser(self, userId):
        seenMovies = {}
        unseenMovies = {}
        matrix = self.matrix
        for item, users in matrix.items():
            if userId in users.keys(): seenMovies[item] = users
            else: unseenMovies[item] = users
        # print(len(seenMovies))
        return (seenMovies, unseenMovies)

    def item_based_recommender(self, target_user_idx):
        matrix = self.matrix
        recommendations = []
        # Compute the similarity between  the target user and each other user in the matrix. 
        # We recomend to store the results in a dataframe (userId and Similarity)
        
        moviesRatingMean = self.calculateRatingsMean()
        seenMovies, unseenMovies = self.findItemsSeenAndNoSeenByUser(target_user_idx)
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
                sim = self.pearsonSimilarity(usersListA, usersListB, moviesRatingMean[kUnseenMovies], moviesRatingMean[kSeenMovies])
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
            
            # Predir el rating de la pel·lícula no vista
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
        # print(' Similarity with item-to-item recommender for user: '+ str(target_user_idx) + ' is ' + str(sim))
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

    target_user_idx = 401
    print('The prediction for user ' + str(target_user_idx) + ':')

    itemToItem = ItemToItem(ratings_train, dataset["movies.csv"], users_idy) 

    recommendations = itemToItem.item_based_recommender(target_user_idx)
    itemToItem.printTopRecommendations()
    itemToItem.validation(ratings_val, target_user_idx)
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")
