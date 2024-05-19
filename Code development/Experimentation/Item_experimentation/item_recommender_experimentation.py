import os
import sys
import time 

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
class_version_dir = os.path.join(code_development_dir, "Class version")
item_dir = os.path.join(code_development_dir, "Experimentation/Item_experimentation")
sys.path.append(code_development_dir)
sys.path.append(class_version_dir)

from utils import *
from Item_based_recommender import item_based_recommender as item # type: ignore


if __name__ == "__main__":
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = split_users(dataset["ratingsSmall_Small.csv"], val_movies)
    
    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    movies = dataset["movies.csv"]

    start = time.time()
    print("Start the prediction of item-to-item based recommender ...")

    itemRecommender = item.ItemToItem(ratings_train, movies, users_idy)
    itemSim = []
    countSim = 0

    for userId in users_idy:
        itemRecommender.item_based_recommender(userId)
        sim = itemRecommender.validation(ratings_val, userId)
        countSim += sim
        itemSim.append((userId, sim))
        print(' Similarity with item-to-item recommender for user: '+ str(userId) + ' is ' + str(sim))

    itemDF = pd.DataFrame(itemSim, columns=['userId', 'itemSim'])
    path = item_dir + '/itemSimSmall_Small.csv'
    itemDF.to_csv(path, index=False)
    
    countSimAverage = countSim / len(users_idy)

    end = time.time()

    print("End the prediction of item based recommender")
    print("The prediction has an average similarity of: " + str(countSimAverage))
    print("The execution time: " + str(end-start) + " seconds")
