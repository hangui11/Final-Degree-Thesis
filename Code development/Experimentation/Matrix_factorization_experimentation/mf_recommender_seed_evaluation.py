import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development\\Class version")
from utils import *
from Matrix_factorization_based_recommender import matrix_factorization_based_recommender as mf # type: ignore
import time 
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
    print("Start the prediction of matrix factorization based recommender ...")

    seeds = [60054000, 42, 243, 1119, 1254392, 3659265380, 2186473283, 172583022, 404699780, 770913450, 6934690, 11680057, 5476520, 9101307, 8657548]
    
    for seed in seeds:
        np.random.seed(seed)
        mfRecommender = mf.MatrixFactorization(ratings_train, movies, users_idy)
        mfRecommender.matrix_factorization()
        countSim = 0
        mfSim = []
        for userId in users_idy:
            mfRecommender.getRecommendations(userId)
            sim = mfRecommender.validation(ratings_val, userId)
            countSim += sim
            mfSim.append((userId, sim))

        mfDF = pd.DataFrame(mfSim, columns=['userId', 'mfSim'])
        path = r'C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Matrix_factorization_experimentation\mfSim('+str(seed)+').csv'
        mfDF.to_csv(path, index=False)
        
        countSimAverage = countSim / len(users_idy)

        end = time.time()
        
        print("End the prediction of matrix factorization based recommender")
        print("The prediction with seed " + str(seed) + " has an average similarity of: " + str(countSimAverage))
        print()

    print("The execution time: " + str(end-start) + " seconds")
