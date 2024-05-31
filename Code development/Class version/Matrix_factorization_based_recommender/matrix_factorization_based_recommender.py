import sys
import os
import numpy as np
import time
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(code_development_dir)

from utils import *

'''
Always need to try update each latent (K) value of P and Q to better predict the model, in this script we use gradient descent to compute and predict the model.

Gradient descent is frequently used as an optimization algorithm in the ML field to learn the latent factors of users and elements. 
Alpha (λ) and beta (β) are regularization parameters commonly used in gradient descent to prevent overfitting and improve the ability of generalization of the model.
It is a very basic model compared to other more complex ones, as it converges with less performance, but we cannot claim that it has a lower accuracy than the others
(Adam, RMSProp, AdaGRAD). In our case, we are using a stochastic gradient descent.

Alpha (α): This is the learning rate, a hyperparameter that controls the size of the steps taken in each iteration during the optimization process.
A higher learning rate can make the algorithm converge faster, but it can also lead to oscillations or divergence in convergence. On the other hand,
a low learning rate may lead to slower convergence, but may be more stable.

Beta (β): This is the regularization parameter, which controls the strength of the regularization applied to the model parameters to avoid overfitting.
The L2 regularization, which is used here, adds a penalty term to the loss function that penalizes large values of the model parameters.
model. A higher value of beta increases the strength of the regularization, which can help prevent overfitting, but it can also make the model
too conservative.

Momentum: It is a hyperparameter that controls the influence of the momentum and smooth the updates of the model parameters during training. This
momentum term is calculated by multiplying the current gradient by a momentum value and adding it to the cumulative change of the parameters 
in previous steps, which it stores in a velocity variable.

The L2 regularization is used to avoid over-fitting during model training.
In summary, the regularization term (β) helps to control the complexity of the model and prevent overfitting by penalizing large valuesof the model parameters
during training of the matrix factorization algorithm. This leads to a model that generalizes better to unseen data and is less prone to overfitting.

Regularization Term= β/2 ∑k=1 (P[i][k]^2 + Q[k][j]^2)

'''


class MatrixFactorization:
    '''
    This method initializes the MatrixFactorization class with the necessary data and parameters,
    preparing it for the KNN hybrid algorithm.
     - self.topK: Stores the number of top recommendations to be generated
     - self.movies: Stores a dataframe with the movies information
     - self.users: Stores a list with the user IDs
     - self.ratings: Stores a dataframe with the ratings information
     - self.featureSize: Stores the number of latent features to be used in the model
    '''
    def __init__(self, ratings_train, movies, users_idy, k=5, featureSize=8) -> None:
        self.movies = movies
        self.users = sorted(users_idy)
        self.topK = k
        self.ratings = ratings_train
        self.featureSize = featureSize
         
    '''
    This function finds the index of an element in its list
    '''
    def findListIndex(self, moviesList, value):
        for i in range(len(moviesList)):
            if (moviesList[i] == value): return i
        return -1

    '''
    This method finds the unseen movies for a given user
    '''
    def findUnseenMoviesByUser(self, target_user_idx):
        ratings = self.ratings
        movies = self.movies['movieId']

        ratingsUser = ratings.loc[(ratings['userId'] == target_user_idx)]
        seenMoviesList = ratingsUser[['movieId']].values.tolist()
        seenMovies = [item for sublist in seenMoviesList for item in sublist]
        unseenMovies = [movie for movie in movies if movie not in seenMovies]

        return unseenMovies

    '''
    This method generates a matrix of users and movies interactions, which some value of rating will be 0
    if the user has not rated the movie.
    '''
    def generate_users_items_matrix(self):
        movies = sorted(self.movies["movieId"].tolist())
        ratings = self.ratings
        # Generate a empty matrix of users and movies interactions
        m = [[0 for i in range(len(movies))] for j in range(len(self.users))]
        for i in self.users:
            # Find all movies rated by the user
            ratingsUser = ratings.loc[(ratings['userId'] == i)]
            data = ratingsUser[['movieId', 'rating']].values.tolist()

            for j in data:
                index = self.findListIndex(movies, j[0])
                userIdx = self.findListIndex(self.users, i)
                m[userIdx][index] = (j[1])

        self.matrix = np.array(m)
        return self.matrix

    '''
    This method compute the matrix factorization model using gradient descent with momentum algorithm
    '''
    def matrix_factorization(self, iterations=10000, alpha=0.0002, beta=0.2, momentum=0.9):
        self.matrix = self.generate_users_items_matrix()
        R = self.matrix

         # Number of users and items
        row, column = R.shape
        # Generate user feature and item feature
        K = self.featureSize        
        # P = np.random.rand(row, K)
        # Q = np.random.rand(K, column)

        # Mean of 3 and Desviation of 2 
        P = np.random.normal(3, 2, size=(row, K))
        Q = np.random.normal(3, 2, size=(K, column))
        
        # Same as matrix P initialized all values set to 0
        velocity_P = np.zeros_like(P)
        # Same as matrix Q initialized all values set to 0
        velocity_Q = np.zeros_like(Q)

        loss_previous = math.inf
        print('Start the MF MODEL computation .....')
        for iteration in range(iterations):
            # Total error of the model in the previous iteration
            e = R - np.dot(P, Q)
            
            # Compute the gradient of the matrix P and Q using the previous error and beta values to regularize the model
            gradient_P = 2*np.dot(e, Q.T) - beta * P
            gradient_Q = 2*np.dot(P.T, e) - beta * Q

            # Normalize the gradients to avoid the explosion of the gradients
            mean_gradient_P = np.mean(gradient_P)
            std_gradient_P = np.std(gradient_P)
            normalized_gradient_P = (gradient_P - mean_gradient_P) / std_gradient_P

            mean_gradient_Q = np.mean(gradient_Q)
            std_gradient_Q = np.std(gradient_Q)
            normalized_gradient_Q = (gradient_Q - mean_gradient_Q) / std_gradient_Q
            
            # Using momentum to accelerate the updates of the model parameters (P and Q)
            velocity_P = velocity_P * momentum + alpha * normalized_gradient_P
            velocity_Q = velocity_Q * momentum + alpha * normalized_gradient_Q

            P += velocity_P
            Q += velocity_Q
            
            # Check for NaN values in the model parameters and replace them with 0
            P = np.nan_to_num(P)
            Q = np.nan_to_num(Q)
            
            # Compute the total error of the model using the regularization term (beta)
            loss = np.sum(e ** 2)/len(e) + (beta/2) * (np.sum(P ** 2) + np.sum(Q ** 2))
            if loss < 0.001:
                break

            if abs(loss-loss_previous) < 1e-3:
                print('The loss respect previous loss is small than 0.001 at the iteration: ' + str(iteration))
                break
            else: loss_previous = loss
            
        self.P = P
        self.Q = Q
        self.R = np.dot(self.P, self.Q)
        return P, Q

    '''
    This method generates the recommendations for a given user
    '''
    def getRecommendations(self, target_user_idx):
        nR = self.R
        user_idx = self.findListIndex(self.users, target_user_idx)
        predictUserRating = list(nR[user_idx])
        movies_idx = sorted(self.movies["movieId"].tolist())
        unseenMovies = self.findUnseenMoviesByUser(target_user_idx)

        recommendations = []
        for i in unseenMovies:
            idx = self.findListIndex(movies_idx, i)
            recommendations.append((i, predictUserRating[idx]))

        # Sort recommendations in descending order
        recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)
        self.recommendations = recommendations
        return recommendations
    
    '''
    This function prints the top recommendations generated by the matrix factorization model
    '''
    def printTopRecommendations(self):
        for recomendation in self.recommendations[:self.topK]:
            rec_movie = self.movies[self.movies["movieId"]  == recomendation[0]]
            print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    '''
    Method to compute the similarity between predictions and the validation dataset,
    which is the same as the similarity between the validation movies genres and the recommended movies genres
    '''
    def validation(self, ratings_val, target_user_idx):
        # Validation
        matrixmpa_genres, validationMoviesGenress = validationMoviesGenres(self.movies, ratings_val, target_user_idx)

        topMoviesUser = list(list(zip(*self.recommendations[:self.topK]))[0])
        recommendsMoviesUser = matrixmpa_genres.loc[topMoviesUser]
        
        # Compute the similarity between the validation movies genres and the recommended movies genres
        sim = cosinuSimilarity(validationMoviesGenress, recommendsMoviesUser)
        # print(' Similarity with matrix factorization recommender: ' + str(sim))
        return sim

if __name__ == "__main__":

    # Set the seed for reproducibility
    number = 9101307 
    np.random.seed(number)

    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = split_users(dataset["ratings.csv"], val_movies)
    
    # Create matrix between user and movies 
    movies_idx = sorted(dataset["movies.csv"]["movieId"].tolist())
    users_idy = sorted(list(set(ratings_train["userId"].values)))
    
    start = time.time()
    target_user_idx = 1
    print('The prediction for user ' + str(target_user_idx) + ':')

    matrixFactorizations = MatrixFactorization(ratings_train, dataset["movies.csv"], users_idy)
    nP, nQ = matrixFactorizations.matrix_factorization()
    
    recommendations = matrixFactorizations.getRecommendations(target_user_idx)
    # print(recommendations)
    matrixFactorizations.printTopRecommendations()
    matrixFactorizations.validation(ratings_val, target_user_idx)

    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")
