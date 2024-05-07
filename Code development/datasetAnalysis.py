import pandas as pd
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    movies = dataset['movies.csv']
    ratings = dataset['ratings.csv']

    
    print('Sneak peak into the ratings dataset:\n\n', ratings.head(), '\n')
    print('Number of ratings: ', ratings.shape[0])
    print('Numver of users: ', ratings['userId'].unique().shape[0])
    print('Number of users who rated at least one movie: ', ratings['userId'].nunique())
    print('Number of movies with at least one rating: ', ratings['movieId'].nunique())
    print('Max rating: ', ratings['rating'].max(), ' Min rating: ', ratings['rating'].min())
    print('Numbers of ratings < 3: ', ratings[ratings['rating'] < 3].shape[0])
    print('Numbers of ratings >= 3: ', ratings[ratings['rating'] >= 3].shape[0])
    print('\n==========================================\n')
    print('Sneak peak into the movies dataset:\n\n', movies.head(), '\n')
    print('Number of movies: ', movies.shape[0])

    print('\n==========================================\n')
    interaction_matrix_size = ratings['userId'].nunique() * ratings['movieId'].nunique()
    interaction_count = ratings.shape[0]
    sparsity = 1 - (interaction_count / interaction_matrix_size)
    print('Sparsity of the interaction matrix: ', sparsity)
    print('\n==========================================\n')

    ratings_expanded = pd.merge(ratings, movies, on='movieId', how='inner')
    ratings_stats = pd.DataFrame(ratings_expanded.groupby('title')['rating'].mean())
    ratings_stats['rating_count'] = pd.DataFrame(ratings_expanded.groupby('title').count()['rating'])
    ratings_stats = ratings_stats.sort_values(by='rating_count', ascending=False)
    print(ratings_stats)
    print('Movies with at most 10 ratings rate by users: ', ratings_stats[ratings_stats['rating_count'] <= 10].shape[0]) 
    print('\n==========================================\n')

    ratings_expanded = pd.merge(ratings, movies, on='movieId', how='inner')
    ratings_stats = pd.DataFrame(ratings_expanded.groupby('userId')['rating'].mean())
    ratings_stats['rating_count'] = pd.DataFrame(ratings_expanded.groupby('userId').count()['rating'])
    ratings_stats = ratings_stats.sort_values(by='rating_count', ascending=False)
    print(ratings_stats)
    print('Users rated at most 100 movies: ', ratings_stats[ratings_stats['rating_count'] <= 100].shape[0]) 




