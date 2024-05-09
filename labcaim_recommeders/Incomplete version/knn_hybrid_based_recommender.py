import pandas as pd
import utils as ut
import user_based_recommender as user
import item_based_recommender as item


def knn_hybrid_based_recommender(usersRecommendations, itemsRecommendations):
    # Provide the code for the hybrid recommender here. This function should return 
    # a list of the top most viewed films according to the combination of user-to-user and item-to-item similarity.
    # We recommend to use pandas library to merge the two recommendation lists and calculate the metric combination.
    # @TODO
    recommendations = []


    return recommendations


if __name__ == "__main__":
    
    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)

    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))

    target_user_idx = 1
    
    ## Remember to compute the user-to-user similarity and item-to-item similarity before merging the recommendations
    # user-to-user similarity
    user_matrix = user.generate_m(users_idy, ratings_train)
    recommendationsUser = user.user_based_recommender(target_user_idx, user_matrix)

    # item-to-item similarity
    items_matrix = item.generate_m(movies_idx, ratings_train)
    recommendationsItem = item.item_based_recommender(target_user_idx, items_matrix)
     
    recommendations = knn_hybrid_based_recommender(recommendationsUser, recommendationsItem)

    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    
     
    








