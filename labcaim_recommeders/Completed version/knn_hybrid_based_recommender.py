import pandas as pd
import numpy as np
import utils as ut
import user_based_recommender as user
import item_based_recommender as item

# Apply combination metric to combine user-to-user and item-to-item similarity
def knn_hybrid_based_recommender(usersRecommendations, itemsRecommendations):
    ## method 1: using pandas
    # recommenderUsers = pd.DataFrame(usersRecommendations, columns=['movieId', 'similarityUser'])
    # recommenderItems = pd.DataFrame(itemsRecommendations, columns=['movieId', 'similarityItem'])
    # recommenderHybrid = recommenderUsers.merge(recommenderItems, how='inner', on='movieId')
    # recommenderHybrid['metricCombination'] = (recommenderHybrid['similarityUser'] * recommenderHybrid['similarityItem']) * (((recommenderHybrid['similarityUser'] - recommenderHybrid['similarityItem']) **2) **0.5)
    # recommendations = [tuple(x) for x in recommenderHybrid[['movieId', 'metricCombination']].to_records(index=False)]

    ## method 2: using dict
    recommenderUsers = {movieId: similarity for movieId, similarity in usersRecommendations}
    recommenderItems = {movieId: similarity for movieId, similarity in itemsRecommendations}

    common_movies = set(recommenderUsers.keys()) & set(recommenderItems.keys())
    recommendations = []
    for i in common_movies:
        userRecommend = recommenderUsers[i]
        itemRecommend = recommenderItems[i]
        recommendations.append((i, (userRecommend * itemRecommend) * (np.sqrt((userRecommend - itemRecommend)**2))))
    
    recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)
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
    
    # user-to-user similarity
    user_matrix = user.generate_m(users_idy, ratings_train)
    recommendationsUser = user.user_based_recommender(target_user_idx, user_matrix)

    # item-to-item similarity
    items_matrix = item.generate_m(movies_idx, ratings_train)
    recommendationsItem = item.item_based_recommender(target_user_idx, items_matrix)
    
    # combine user-to-user and item-to-item similarity
    recommendations = knn_hybrid_based_recommender(recommendationsUser, recommendationsItem)

    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    
     
    








