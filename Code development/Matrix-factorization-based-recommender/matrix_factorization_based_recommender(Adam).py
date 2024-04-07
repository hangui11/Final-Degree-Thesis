'''
Optimizer is a crucial element that fine-tunes a neuronal network's parameters during the training. Minimize the model's
error or loss function. Has many types of optimizer, the basic one is Gradient Descent, the other are more complex.
The Adam optimizer is one of the most popular optimizer in DL, and can converge the model with a big performance and
efficiency. Ensuring a more effective and quicker path to the lowest point, which represents the least loss in machine learning.
Adam tweaks the gradient descent method by considering the moving average of the first and second-order moments of the gradient.
This allows it to adapt the learning rates for each parameter intelligently. 
The vector m is intended to store the moving average of the gradients, while v keeps track of the moving average of the squared gradients. 
m0​=0 (Initial first-moment vector)
v0​=0 (Initial second-moment vector)
the gradient is the derivate of the objective function
'''
# def matrix_factorization1(R, P, Q, K, iterations=20000, learning_rate=0.0001, beta=0.1):
#     Q = Q.T
    
    
#     # Inicializar los parámetros del algoritmo Adam
#     beta1 = 0.9
#     beta2 = 0.999
#     epsilon = 1e-8
#     mP = np.zeros(P.shape)
#     vP = np.zeros(P.shape)
#     mQ = np.zeros(Q.shape)
#     vQ = np.zeros(Q.shape)

#     for iteration in range(iterations):
#         # Calcular el error
#         e = R - np.dot(P, Q)

#         # Calcular los gradientes para P
#         gradient_P = np.dot(e, Q.T) - beta * P

#         # Calcular los gradientes para Q
#         gradient_Q = np.dot(P.T, e) - beta * Q

#         # Actualizar m y v para P
#         mP = beta1 * mP + (1 - beta1) * gradient_P
#         vP = beta2 * vP + (1 - beta2) * gradient_P ** 2

#         # Actualizar m y v para Q
#         mQ = beta1 * mQ + (1 - beta1) * gradient_Q
#         vQ = beta2 * vQ + (1 - beta2) * gradient_Q ** 2

#         # Calcular los sesgos corregidos
#         mP_hat = mP / (1 - beta1 ** (iteration + 1))
#         vP_hat = vP / (1 - beta2 ** (iteration + 1))
#         mQ_hat = mQ / (1 - beta1 ** (iteration + 1))
#         vQ_hat = vQ / (1 - beta2 ** (iteration + 1))

#         # Actualizar los parámetros P y Q
#         P += learning_rate * mP_hat / (np.sqrt(vP_hat) + epsilon)
#         Q += learning_rate * mQ_hat / (np.sqrt(vQ_hat) + epsilon)
        
#         # Calcular la función de pérdida
#         loss = np.sum(e ** 2) + (beta/2) * (np.sum(P ** 2) + np.sum(Q ** 2))
        
#         if loss < 0.001:
#             print(iteration)
#             break           
#     return P, Q

import utils as ut
import numpy as np
import time
import naive_recommender as nav
import torch
import torch.nn as nn

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

def generate_users_items_matrix(movies, ratings, target_user_idx):
    m = [[0 for i in range(len(movies))]]
    
    ratingsUser = ratings.loc[(ratings['userId'] == target_user_idx)]
    data = ratingsUser[['movieId', 'rating']].values.tolist()

    for j in data:
        index = findMovieIndex(movies, j[0])
        m[0][index] = (j[1])

    return np.array(m)


if __name__ == "__main__":
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)
    
    # Create matrix between user and movies 
    movies_idx = sorted(dataset["movies.csv"]["movieId"].tolist())
    users_idy = sorted(list(set(ratings_train["userId"].values)))
    
    start = time.time()
    target_user_idx = 172
    print('The prediction for user: ' + str(target_user_idx))
    m = generate_users_items_matrix(movies_idx, ratings_train, target_user_idx)

    # NUMBER OF USERS AND NUMBER OF ITEMS
    num_users, num_items = m.shape
    # NUMBER OF FEATURES
    K = 10
    learning_rate = 0.0002
    epochs = 10000
    unseenMovies = findUnseenMoviesByUser(ratings_train, target_user_idx, movies_idx)

    # Initialize the P and Q parameters with normal random values
    P = nn.Parameter(torch.randn(num_users, K, requires_grad=True))
    Q = nn.Parameter(torch.randn(num_items, K, requires_grad=True))
    # Compute the loss function
    loss_fn = nn.MSELoss()
    # Compute the optimizer
    optimizer = torch.optim.Adam([P, Q], lr=learning_rate)

    m_tensor = torch.FloatTensor(m)
    for epoch in range(epochs):
        R_pred = torch.matmul(P, Q.t())
        e = loss_fn(m_tensor, R_pred)
        optimizer.zero_grad()
        e.backward()
        optimizer.step()
        #if epoch % 100 == 0:
            # print(f'Epoch {epoch + 1}, Loss: {e.item()}')
    
    R_pred = torch.matmul(P, Q.t()).detach().numpy()

    predictUserRating = R_pred[0]
    recommendations = []

    for i in unseenMovies:
        idx = findMovieIndex(movies_idx, i)
        recommendations.append((i, predictUserRating[idx]))
        
    recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)

    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    # Validation
    matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])
    validationMovies = ratings_val['movieId'].loc[ratings_val['userId'] == target_user_idx].values.tolist()
    validationMoviesGenres = matrixmpa_genres.loc[validationMovies]

    # matrix factorization based recommender
    topMoviesUser = list(list(zip(*recommendations[:5]))[0])
    recommendsMoviesGenresItem = matrixmpa_genres.loc[topMoviesUser]
    # sim entre matriu genere amb recomanador matrix factorization
    sim = ut.cosinuSimilarity(validationMoviesGenres, recommendsMoviesGenresItem)
    print(' Similarity with matrix factorization recommender: ' + str(sim))
    print()

    # naive recommender
    topMovieSystem = nav.naive_recommender(ratings_val, dataset["movies.csv"], 5)
    topMovieSystem = topMovieSystem['movieId'].values.tolist()
    for recomendation in topMovieSystem:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
    recommendsMoviesGenresSystem = matrixmpa_genres.loc[topMovieSystem]
    # sim entre matriu genere amb recomanador sistema
    sim2 = ut.cosinuSimilarity(validationMoviesGenres, recommendsMoviesGenresSystem)
    print(' Similarity with naive recommender: ' + str(sim2))
    
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")

    print(recommendations[:5])

    end = time.time()
    print(end-start)





