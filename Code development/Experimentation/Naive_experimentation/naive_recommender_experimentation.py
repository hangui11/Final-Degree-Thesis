import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development\\Class version")
from utils import *
from Naive_based_recommender import naive_recommender as nav
import time 

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
    print("Start the prediction of naive based recommender ...")

    navRecommender = nav.Naive(ratings_train, movies)
    navRecommender.naive_recommender()
    navSim = []
    countSim = 0
    
    for userId in users_idy:
        sim = navRecommender.validation(ratings_val, userId)
        countSim += sim
        
        navSim.append((userId, sim))

    naiveDF = pd.DataFrame(navSim, columns=['userId', 'navSim'])
    path = 'C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Naive_experimentation\navSim.csv'
    naiveDF.to_csv(path, index=False)
    
    countSimAverage = countSim / len(users_idy)

    end = time.time()

    print("End the prediction of naive based recommender")
    print("The prediction has an average similarity of: " + str(countSimAverage))
    print("The execution time: " + str(end-start) + " seconds")
