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
Intentamos siempre de actualizar cada valor de latente (K) de P i Q para prevenir de mejor manera el modelo, en este capítulo usamos el descenso de gradiente para
computar y prevenir el nuestro modelo, que es rating prediction

El descenso de gradiente se usa frecuentemente como algoritmo de optimización en el campo de ML para aprender los factores latentes de los usuarios y elementos. 
Alpha (λ) y beta (β) son parámetros de regularización comúnmente utilizados en el descenso de gradiente para prevenir el sobreajuste y mejorar la capacidad de
generalización del modelo. Es un modelo muy básico comparado con otros más complejos, ya que converja con menos rendimiento, pero no podemos afirmar que tiene un menor
precisión que los otros (Adam, RMSProp, AdaGRAD). En nuestro caso, estamos usanto un stochastic gradient descent.

Alpha (α): Es la tasa de aprendizaje, un hiperparámetro que controla el tamaño de los pasos que se dan en cada iteración durante el proceso de optimización.
Una tasa de aprendizaje más alta puede hacer que el algoritmo converja más rápido, pero también puede provocar oscilaciones o divergencia en la convergencia.
Por otro lado, una tasa de aprendizaje baja puede llevar a una convergencia más lenta, pero puede ser más estable.

Beta (β): Es el parámetro de regularización, que controla la fuerza de la regularización aplicada a los parámetros del modelo para evitar el sobreajuste.
La regularización L2, que se utiliza aquí, agrega un término de penalización a la función de pérdida que penaliza los valores grandes de los parámetros del
modelo. Un valor más alto de beta aumenta la fuerza de la regularización, lo que puede ayudar a prevenir el sobreajuste, pero también puede hacer que el modelo
sea demasiado conservador.

Momentum: Es un hiperparámetro que controla la influencia del momento y suavizar las actualizaciones de los parámetros del modelo durante el entrenamiento. Este
término de momentum se calcula multiplicando el gradiente actual por un valor de momentum y sumándolo al cambio acumulado de los parámetros en pasos anteriores,
que almacena en una variable de velocidad (velocity).

La regularización L2 se utiliza para evitar el sobreajuste durante el entrenamiento del modelo.
En resumen, el término de regularización (β) ayuda a controlar la complejidad del modelo y a prevenir el sobreajuste al penalizar los valores grandes
de los parámetros del modelo durante el entrenamiento del algoritmo de factorización de matrices. Esto conduce a un modelo que generaliza mejor a datos
no vistos y que es menos propenso al sobreajuste.

Regularization Term= β/2 ∑k=1 (P[i][k]^2 + Q[k][j]^2)

'''


class MatrixFactorization:
    def __init__(self, ratings_train, movies, users_idy, k=5, featureSize=8) -> None:
        self.movies = movies
        self.users = sorted(users_idy)
        self.topK = k
        self.ratings = ratings_train
        self.featureSize = featureSize
         
    def findMovieIndex(self, moviesList, value):
        for i in range(len(moviesList)):
            if (moviesList[i] == value): return i
        return -1

    def findUnseenMoviesByUser(self, target_user_idx):
        ratings = self.ratings
        movies = self.movies['movieId']

        ratingsUser = ratings.loc[(ratings['userId'] == target_user_idx)]
        seenMoviesList = ratingsUser[['movieId']].values.tolist()
        seenMovies = [item for sublist in seenMoviesList for item in sublist]
        unseenMovies = [movie for movie in movies if movie not in seenMovies]

        return unseenMovies

    def generate_users_items_matrix(self):
        movies = sorted(self.movies["movieId"].tolist())
        ratings = self.ratings
        m = [[0 for i in range(len(movies))] for j in range(len(self.users))]
        for i in self.users:
            ratingsUser = ratings.loc[(ratings['userId'] == i)]
            data = ratingsUser[['movieId', 'rating']].values.tolist()

            for j in data:
                index = self.findMovieIndex(movies, j[0])
                m[i-1][index] = (j[1])

        self.matrix = np.array(m)
        return self.matrix

    def matrix_factorization(self, iterations=10000, alpha=0.0002, beta=0.2, momentum=0.9):
        self.matrix = self.generate_users_items_matrix()
        R = self.matrix

         # NUMBER OF USERS AND NUMBER OF ITEMS
        row, column = R.shape
        # Generate user feature and item feature
        K = self.featureSize
        
        # P = np.random.rand(row, K)
        # Q = np.random.rand(K, column)

        # Mean of 3 and Desviation of 2 
        P = np.random.normal(3, 2, size=(row, K))
        Q = np.random.normal(3, 2, size=(K, column))
        
        # Igual que la matriz P inicializado con todos los valores a 0
        velocity_P = np.zeros_like(P)
        # Igual que la matriz Q inicializado con todos los valores a 0
        velocity_Q = np.zeros_like(Q)

        loss_previous = math.inf
        print('Start the MF MODEL computation .....')
        for iteration in range(iterations):
            # Error total del modelo
            e = R - np.dot(P, Q)
            
            # Calcular los gradientes para P
            gradient_P = 2*np.dot(e, Q.T) - beta * P
            # Calcular los gradientes para Q
            gradient_Q = 2*np.dot(P.T, e) - beta * Q

            # Normalización z-score de los gradientes
            mean_gradient_P = np.mean(gradient_P)
            std_gradient_P = np.std(gradient_P)
            normalized_gradient_P = (gradient_P - mean_gradient_P) / std_gradient_P

            mean_gradient_Q = np.mean(gradient_Q)
            std_gradient_Q = np.std(gradient_Q)
            normalized_gradient_Q = (gradient_Q - mean_gradient_Q) / std_gradient_Q
            
            ## Uso de momentum para convergir más rápido
            velocity_P = velocity_P * momentum + alpha * normalized_gradient_P
            velocity_Q = velocity_Q * momentum + alpha * normalized_gradient_Q

            P += velocity_P
            Q += velocity_Q
            
            P = np.nan_to_num(P)
            Q = np.nan_to_num(Q)
            
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

    def getRecommendations(self, target_user_idx):
        nR = self.R
        predictUserRating = list(nR[target_user_idx-1])
        movies_idx = sorted(self.movies["movieId"].tolist())
        unseenMovies = self.findUnseenMoviesByUser(target_user_idx)
        recommendations = []
        
        for i in unseenMovies:
            idx = self.findMovieIndex(movies_idx, i)
            recommendations.append((i, predictUserRating[idx]))
            
        recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)
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
