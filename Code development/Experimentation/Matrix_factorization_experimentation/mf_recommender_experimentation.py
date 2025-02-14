import os
import sys
import time 
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
class_version_dir = os.path.join(code_development_dir, "Class version")
mf_dir = os.path.join(code_development_dir, "Experimentation/Matrix_factorization_experimentation")
sys.path.append(code_development_dir)
sys.path.append(class_version_dir)

from utils import *
from Matrix_factorization_based_recommender import matrix_factorization_based_recommender as mf # type: ignore


if __name__ == "__main__":
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = load_dataset_from_source(path_to_ml_latest_small)

    # Split the dataset into training and validation set
    val_movies = 5
    ratings_train, ratings_val = split_users(dataset["ratings.csv"], val_movies)
    
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    movies = dataset["movies.csv"]

    start = time.time()
    print("Start the prediction of matrix factorization based recommender ...")

    # Set the random seed for reproducibility
    seed = 9101307
    np.random.seed(seed)
    
    # Create the matrix factorization recommender
    mfRecommender = mf.MatrixFactorization(ratings_train, movies, users_idy)
    # Train the model
    mfRecommender.matrix_factorization()
    end = time.time()
    print('MF MODEL Computation time: ' + str(end-start))

    mfSim = []
    countSim = 0

    # Make the prediction of the matrix factorization based recommender for each user in the validation set
    for userId in users_idy:
        mfRecommender.getRecommendations(userId)
        sim = mfRecommender.validation(ratings_val, userId)
        countSim += sim
        mfSim.append((userId, sim))
        print(' Similarity with matrix factorization recommender for user: '+ str(userId) + ' is ' + str(sim))

    # Save the similarity of each user with the matrix factorization based recommender in a csv file
    mfDF = pd.DataFrame(mfSim, columns=['userId', 'mfSim'])
    path = mf_dir + '/mfSim.csv'
    mfDF.to_csv(path, index=False)
        
    countSimAverage = countSim / len(users_idy)

    end = time.time()
        
    print("End the prediction of matrix factorization based recommender")
    print("The prediction has an average similarity of: " + str(countSimAverage))

    print("The execution time: " + str(end-start) + " seconds")
