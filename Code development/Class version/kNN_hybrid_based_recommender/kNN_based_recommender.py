import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development\\Class version")
from utils import *
import numpy as np
import time
from Item_based_recommender import item_based_recommender as item
from User_based_recommender import user_based_recommender as user

class KnnHybrid:
    def __init__(self, movies, k=5) -> None:
        self.movies = movies
        self.topK = k

    def mergeUsersAndItemsRecommendations(self, usersRecommendations, itemsRecommendations):
        ## method 1: using pandas
        # recommenderUsers = pd.DataFrame(usersRecommendations, columns=['movieId', 'similarityUser'])
        # recommenderItems = pd.DataFrame(itemsRecommendations, columns=['movieId', 'similarityItem'])

        # recommenderHybrid = recommenderUsers.merge(recommenderItems, how='inner', on='movieId')
        # recommenderHybrid['productSimilarities'] = recommenderHybrid['similarityUser'] * recommenderHybrid['similarityItem']
        # recommenderHybrid['maxSimilarity'] = recommenderHybrid[['similarityUser', 'similarityItem']].max(axis=1)
        # recommenderHybrid['euclidianDistance'] = ((recommenderHybrid['similarityUser'] - recommenderHybrid['similarityItem']) **2) **0.5

        # recommendations1 = [tuple(x) for x in recommenderHybrid[['movieId', 'productSimilarities']].to_records(index=False)]
        # recommendations2 = [tuple(x) for x in recommenderHybrid[['movieId', 'maxSimilarity']].to_records(index=False)]
        # recommendations3 = [tuple(x) for x in recommenderHybrid[['movieId', 'euclidianDistance']].to_records(index=False)]
        
        ## method 2: using dict
        recommenderUsers = {movieId: similarity for movieId, similarity in usersRecommendations}
        recommenderItems = {movieId: similarity for movieId, similarity in itemsRecommendations}

        common_movies = set(recommenderUsers.keys()) & set(recommenderItems.keys())

        productSimilarities = []
        euclidianDistance = []
        # La combinació permet capturar tant la diferencia com la relació entre els resultats de dos similituds
        # Ja que aquesta combinació pot afavorir la complexitat d'aquest mètode perquè captura una gamma més amplia
        # de les relacions entre els elements (similitat de dos mètodes: user-to-user i item-to-item)
        combinationProductAndEuclidian = []
        for i in common_movies:
            userRecommend = recommenderUsers[i]
            itemRecommend = recommenderItems[i]
            productSimilarities.append((i, userRecommend * itemRecommend))
            euclidianDistance.append((i, np.sqrt((userRecommend - itemRecommend)**2) ))
            combinationProductAndEuclidian.append((i, (userRecommend * itemRecommend) * (np.sqrt((userRecommend - itemRecommend)**2))))

        self.recommendations = sorted(combinationProductAndEuclidian, key=lambda x:x[1], reverse=True)
        return self.recommendations
    
    def printTopRecommendations(self):
        for recomendation in self.recommendations[:self.topK]:
            rec_movie = self.movies[self.movies["movieId"]  == recomendation[0]]
            print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    def validation(self, ratings_val, target_user_idx):
        # Validation
        matrixmpa_genres, validationMoviesGenress = validationMoviesGenres(self.movies, ratings_val, target_user_idx)

        topMoviesUser = list(list(zip(*self.recommendations[:self.topK]))[0])
        recommendsMoviesUser = matrixmpa_genres.loc[topMoviesUser]
        
        # sim entre matriu genere amb recomanador user
        sim = cosinuSimilarity(validationMoviesGenress, recommendsMoviesUser)
        # print(' Similarity with kNN hybrid recommender: ' + str(sim))
        return sim

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

    target_user_idx = 401
    print('The prediction for user ' + str(target_user_idx) + ':')

    start = time.time()
    kNN = KnnHybrid(dataset["movies.csv"])
    
    userToUser = user.UserToUser(ratings_train, dataset["movies.csv"], users_idy)
    usersRecommendations = userToUser.user_based_recommender(target_user_idx)

    itemToItem = item.ItemToItem(ratings_train, dataset["movies.csv"], users_idy)
    itemsRecommendations = itemToItem.item_based_recommender(target_user_idx)

    recommendations = kNN.mergeUsersAndItemsRecommendations(usersRecommendations, itemsRecommendations)
    kNN.printTopRecommendations()
    kNN.validation(ratings_val, target_user_idx)
    
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")
