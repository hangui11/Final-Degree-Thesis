import os
import sys
import time 
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
class_version_dir = os.path.join(code_development_dir, "Class version")
ncf_dir = os.path.join(code_development_dir, "Experimentation/Neuronal_colaborative_filter_experimentation")
sys.path.append(code_development_dir)
sys.path.append(class_version_dir)

from utils import *
from Neuronal_colaborative_filter_based_recommender import neuronal_colaborative_filter_based_recommender as ncf # type: ignore


if __name__ == "__main__":
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = split_users(dataset["ratingsSmall_Small.csv"], val_movies)
    
    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = sorted(list(set(ratings_train["userId"].values)))
    movies = dataset["movies.csv"]

    start = time.time()
    print("Start the prediction of neuronal colaborative filter based recommender ...")

    seed = 9101307
    torch.manual_seed(seed)
    np.random.seed(seed)
    ncfRecommender = ncf.NeuronalColaborativeFilter(len(users_idy), len(movies_idx), ratings_train, movies)
    ncfRecommender.trainingModel(lr=1e-3, wd=1e-4, max_epochs = 50, batch_size = 64,  early_stop_epoch_threshold = 5)
    ncfRecommender.evaluateModel(ratings_val, batch_size = 64, users = users_idy)
    end = time.time()
    print('NCF MODEL Computation time: ' + str(end-start))

    ncfSim = []
    countSim = 0

    for userId in users_idy:
        ncfRecommender.predictUnseenMoviesRating(userId, users_idy)
        sim = ncfRecommender.validation(ratings_val, userId)
        countSim += sim
        ncfSim.append((userId, sim))
        print(' Similarity with neuronal colaborative filter recommender for user: '+ str(userId) + ' is ' + str(sim))

    ncfDF = pd.DataFrame(ncfSim, columns=['userId', 'ncfSim'])
    path = ncf_dir + '/ncfSimSmall_Small.csv'
    ncfDF.to_csv(path, index=False)
        
    countSimAverage = countSim / len(users_idy)

    end = time.time()
        
    print("End the prediction of neuronal colaborative filter based recommender")
    print("The prediction has an average similarity of: " + str(countSimAverage))

    print("The execution time: " + str(end-start) + " seconds")
