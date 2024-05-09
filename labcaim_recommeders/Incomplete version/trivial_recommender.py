import pandas as pd 
import utils as ut

def trivial_recommender(ratings: object, movies:object, k: int = 5) -> list: 
    # Provide the code for the trivial recommender here. This function should return 
    # the list of the top most viewed films according to the ranking (sorted in descending order).
    # Consider using the utility functions from the pandas library.
    most_seen_movies = []
    
    return most_seen_movies[:5]


if __name__ == "__main__":
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]

    # Split the dataset into training and validation sets
    ratings_train, ratings_val = ut.split_users(ratings, 5)
    
    # Test the recommender
    recommendations = trivial_recommender(ratings, movies)
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))
    print('\n',recommendations)

