import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
from utils import *
import numpy as np
import time
import math
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


def findMovieIndex(moviesList, value):
    for i in range(len(moviesList)):
        if (moviesList[i] == value): return i
    return -1


def findUnseenMoviesByUser(ratings, target_user_idx, movies):
    ratingsUser = ratings.loc[(ratings['userId'] == target_user_idx)]
    seenMoviesList = ratingsUser[['movieId']].values.tolist()
    seenMovies = [item for sublist in seenMoviesList for item in sublist]
    unseenMovies = [movie for movie in movies if movie not in seenMovies]

    return unseenMovies

def generate_users_items_matrix(movies, ratings, users):
    m = [[0 for i in range(len(movies))] for j in range(len(users))]
    
    for i in users:
        ratingsUser = ratings.loc[(ratings['userId'] == i)]
        data = ratingsUser[['movieId', 'rating']].values.tolist()

        for j in data:
            index = findMovieIndex(movies, j[0])
            m[i-1][index] = (j[1])

    return np.array(m)

def matrix_factorization(R, P, Q, K,  iterations=10000, alpha=0.0001, beta=0.1, momentum=0.9):
    Q = Q.T
    # Igual que la matriz P inicializado con todos los valores a 0
    velocity_P = np.zeros_like(P)
    # Igual que la matriz Q inicializado con todos los valores a 0
    velocity_Q = np.zeros_like(Q)

    loss_previous = math.inf
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
        
        loss = np.sum(e ** 2) + (beta/2) * (np.sum(P ** 2) + np.sum(Q ** 2))

        
        if loss < 0.001:
            print(iteration)
            break

        # if abs(loss-loss_previous) < 10e-4:
        #     P1 = P
        #     Q1 = Q
        if abs(loss-loss_previous) < 10e-3:
            # print(iteration)
            break
        else: loss_previous = loss
        
    return P, Q

def getRecommendations(ratings_train, target_user_idx, movies_idx, predictUserRating):
    unseenMovies = findUnseenMoviesByUser(ratings_train, target_user_idx, movies_idx)
    recommendations = []
    
    for i in unseenMovies:
        idx = findMovieIndex(movies_idx, i)
        recommendations.append((i, predictUserRating[idx]))
        
        
    recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)
    
    return recommendations

if __name__ == "__main__":
    np.random.seed(42)
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
    target_user_idx = 11
    print('The prediction for user ' + str(target_user_idx) + ':')

    m = generate_users_items_matrix(movies_idx, ratings_train, users_idy)

    # NUMBER OF USERS AND NUMBER OF ITEMS
    row, column = m.shape
    # NUMBER OF FEATURES
    K = 8
    P = np.random.rand(row, K)
    Q = np.random.rand(column, K)
    
    nP, nQ = matrix_factorization(m, P, Q, K)
    nR = np.dot(nP, nQ)
    predictUserRating = list(nR[target_user_idx-1])

    recommendations = getRecommendations(ratings_train, target_user_idx, movies_idx, predictUserRating)
    
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    # Validation
    matrixmpa_genres, validationMoviesGenres = validationMoviesGenres(dataset["movies.csv"], ratings_val, target_user_idx)

    # matrix factorization based recommender
    topMoviesUser = list(list(zip(*recommendations[:5]))[0])
    recommendsMoviesGenresItem = matrixmpa_genres.loc[topMoviesUser]
    # sim entre matriu genere amb recomanador matrix factorization
    sim = cosinuSimilarity(validationMoviesGenres, recommendsMoviesGenresItem)
    print(' Similarity with matrix factorization recommender: ' + str(sim))

    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")

    # print(recommendations[:5])






