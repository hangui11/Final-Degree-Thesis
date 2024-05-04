import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development\\Class version")
from utils import *
from Neuronal_colaborative_filter_based_recommender import neuronal_colaborative_filter_based_recommender as ncf # type: ignore
import time 
import torch
import numpy as np

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
    movies = dataset["movies.csv"]

    start = time.time()
    print("Start the prediction of neuronal colaborative filter based recommender ...")

    seed = 9101307
    torch.manual_seed(seed)
    np.random.seed(seed)
    ncfRecommender = ncf.NeuronalColaborativeFilter(len(users_idy), len(movies_idx), ratings_train, movies)
    ncfRecommender.trainingModel(lr=1e-3, wd=1e-5, max_epochs = 50, batch_size = 64,  early_stop_epoch_threshold = 5)
    ncfRecommender.evaluateModel(ratings_val, batch_size = 64)
    end = time.time()
    print('NCF MODEL Computation time: ' + str(end-start))

    ncfSim = []
    countSim = 0

    for userId in users_idy:
        ncfRecommender.predictUnseenMoviesRating(userId)
        sim = ncfRecommender.validation(ratings_val, userId)
        countSim += sim
        ncfSim.append((userId, sim))
        print(' Similarity with neuronal colaborative filter recommender for user: '+ str(userId) + ' is ' + str(sim))

    ncfDF = pd.DataFrame(ncfSim, columns=['userId', 'ncfSim'])
    path = r'C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Neuronal_colaborative_filter_experimentation\ncfSim2.csv'
    ncfDF.to_csv(path, index=False)
        
    countSimAverage = countSim / len(users_idy)

    end = time.time()
        
    print("End the prediction of neuronal colaborative filter based recommender")
    print("The prediction has an average similarity of: " + str(countSimAverage))

    print("The execution time: " + str(end-start) + " seconds")
