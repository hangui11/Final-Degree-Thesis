import os
import pandas as pd
import numpy as np
import utils as ut
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
dataset_dir = os.path.join(parent_dir, "ml-latest-small")

def createSmallDataset(ratings):
    ratings_stats = pd.DataFrame(ratings.groupby('userId')['rating'].mean())
    ratings_stats['rating_count'] = pd.DataFrame(ratings.groupby('userId').count()['rating'])
    ratings_stats = ratings_stats.sort_values(by='rating_count', ascending=False)
    users = ratings_stats[ratings_stats['rating_count'] <= 900].index.tolist()
    ratings_small = ratings[ratings['userId'].isin(users)]

    return ratings_small


if __name__ == '__main__':
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    movies = dataset['movies.csv']
    ratings = dataset['ratings.csv']


    small_ratings = createSmallDataset(ratings)
    path = dataset_dir + '/ratingsSmall.csv'

    small_ratings.to_csv(path, index=False)