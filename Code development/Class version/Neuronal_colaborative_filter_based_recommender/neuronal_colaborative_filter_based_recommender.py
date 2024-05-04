import sys
sys.path.append("C:\\Users\\usuario\\Desktop\\FIB\\Final-Degree-Thesis\\Code development")
from utils import *
import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

# 243 933 42 
number = 9101307
print(number)
torch.manual_seed(number)
np.random.seed(number)

class DatasetBatchIterator:
    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X = self.X[index]
            Y = self.Y[index]
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(X.shape[0] / batch_size))
        self._current = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        X_batch = torch.LongTensor(self.X[k*bs:(k+1)*bs])
        Y_batch = torch.FloatTensor(self.Y[k*bs:(k+1)*bs])
        return X_batch, Y_batch.view(-1, 1)

class NeuronalColaborativeFilter(nn.Module):

    def __init__(self, user_count, item_count, ratings_train, movies, embedding_size=64, hidden_layers=(128,64,32,16), dropout_rate=None, output_range=(1,5), k=5):
        super().__init__()
        self.movies = movies
        self.ratings = ratings_train
        self.ratings_x = ratings_train[['userId', 'movieId']]
        self.ratings_x.loc[:,'userId'] = self.ratings_x['userId'] - 1
        movies_idx = self.findIdx(self.ratings_x['movieId'].values.tolist())
        self.ratings_x.loc[:,'movieId'] = movies_idx
        self.ratings_y = ratings_train['rating'].astype(np.float32)
        
        self.topK = k

        ## Initialize the GPU device to compute
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        ## Initialize embedding hash sizes
        self.user_hash_size = user_count
        self.item_hash_size = item_count

        ## Initialize the model architecture components
        self.user_embedding = nn.Embedding(user_count, embedding_size)
        self.item_embedding = nn.Embedding(item_count, embedding_size)

        ## Generate a MLP
        self.MLP = self.__genMLP(embedding_size, hidden_layers, dropout_rate)

        # The probability that an unit of a layer will be closed
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        
        ## Initialize output normalization parameters
        assert output_range and len(output_range) == 2, "output range has to be tuple with two integers"
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[0]-output_range[1]) + 1
        
        self.initParams()

    def __genMLP(self, embedding_size, hidden_layers_units, dropout_rate):
        assert (embedding_size * 2) == hidden_layers_units[0], "First input layers number units must be the twice time of embedding size"

        hidden_layers = []
        input_units = hidden_layers_units[0]

        # Construct the connection between layers
        for num_units in hidden_layers_units[1:]:
            # Connect the inputs_units layers with num_units layers linealy
            hidden_layers.append(nn.Linear(input_units, num_units, device=self.device))
            # Normalize the before layer's activities to stabilize and accelerate training
            hidden_layers.append(nn.BatchNorm1d(num_units, device=self.device))
            # Learn the relation between no connect layers
            hidden_layers.append(nn.ReLU())

            if dropout_rate:
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        # Conect the last layers with the output
        hidden_layers.append(nn.Linear(hidden_layers_units[-1], 1, device=self.device))
        # Reproduce output between 0 and 1 
        hidden_layers.append(nn.Sigmoid())
        
        return nn.Sequential(*hidden_layers)
    
    def initParams(self):
        def weights_init(m):
            if type(m) == nn.Linear:
                '''
                Initialize the weights of a neuronal network layer using Kaiming uniform
                The Kaiming initialization method aims to set the initial weights in such 
                a way that the variance of the outputs of each layer remains approximately
                the same during forward and backward propagation.
                The bias help neuronal network to learn patrons more complex and fitting
                well the inputs
                '''
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.fill_(0.01)

        ## Initialize the value of embeddings with random number between -0.05 and 0.05
        embedding_range = 0.05
        self.user_embedding.weight.data.uniform_(-embedding_range, embedding_range)
        self.item_embedding.weight.data.uniform_(-embedding_range, embedding_range)

        ## Each layer of MLP will initialize according this function 'weights_init'
        self.MLP.apply(weights_init)

    ## This function will be auto invoked
    def forward(self, user_id, item_id):
        ## Access the features of user_id and item_id
        
        user_features = self.user_embedding(user_id)
        item_features = self.item_embedding(item_id)
        
        ## Concat the features of user and item in one feature representation
        x = torch.cat([user_features, item_features], dim=1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        ## The feature will be compute with neuronal network 
        x = self.MLP(x)

        ## Normalize the output between 1 and 5
        normalized_output = x * self.norm_range + self.norm_min
        return normalized_output

    def findIdx(self, movie_ids):
        idx = []
        listMovies = self.movies['movieId'].values.tolist()
        for movie_id in movie_ids:
            if movie_id in listMovies:
                idx.append(listMovies.index(movie_id))
            # else:
            #     print("eroorrrr")
        return idx

    
    def trainingModel(self, lr, wd, max_epochs, early_stop_epoch_threshold, batch_size):
        # self.initParams()
        self.train()

        ## Training loop control parameters
        no_loss_reduction_epoch_counter = 0
        min_loss = np.inf
        
        # Use GPU to run the Neuronal Network
        self.to(self.device)

        ## Loss function: Huber Error -> combination of MAE and MSE
        beta = 0.25
        loss_criterion = nn.SmoothL1Loss(reduction='sum', beta=beta)
        ## Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        ## Training of the model
        print('Start the NCF MODEL training .....')
        
        ## Each epoch update our model
        for epoch in range(max_epochs):
            epoch_loss = 0.0

            for x_batch, y_batch in DatasetBatchIterator(self.ratings_x, self.ratings_y, batch_size=batch_size, shuffle=True):
                x_batch, y_batch = x_batch.to(device=self.device), y_batch.to(device=self.device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self(x_batch[:, 0], x_batch[:, 1])
                    loss = loss_criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
        
            epoch_loss = epoch_loss / len(self.ratings_x)
            print(f'Epoch: {epoch+1}, Loss: {epoch_loss}')

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                no_loss_reduction_epoch_counter = 0
            else:
                no_loss_reduction_epoch_counter += 1
            if no_loss_reduction_epoch_counter >= early_stop_epoch_threshold:
                print(f'Early stop at epoch {epoch+1}')
                break

    def evaluateModel(self, ratings_val, batch_size):
        self.eval()
        ratings_val_x = ratings_val[['userId', 'movieId']]
        ratings_val_x.loc[:,'userId'] = ratings_val_x['userId'] - 1
        movies_idx = self.findIdx(ratings_val_x['movieId'].values.tolist())
        ratings_val_x.loc[:,'movieId'] = movies_idx
        ratings_val_y = ratings_val[['rating']]
        groud_truth, predictions = [], []
        with torch.no_grad():
            for x_batch, y_batch in DatasetBatchIterator(ratings_val_x, ratings_val_y, batch_size=batch_size, shuffle=False):
                x_batch, y_batch = x_batch.to(device=self.device), y_batch.to(device=self.device)
                
                outputs = self(x_batch[:, 0], x_batch[:, 1])
                groud_truth.extend(y_batch.tolist())
                predictions.extend(outputs.tolist())
        groud_truth = np.array(groud_truth).ravel()
        predictions = np.array(predictions).ravel()
        RMSE = np.sqrt(mean_squared_error(groud_truth, predictions))
        print(f'RMSE: {RMSE}')

    ## Prediction of rating for a movie
    def predictRatingMovie(self, user_id, movie_id):
        self.eval()
        device = next(self.parameters()).device
        user_id_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        movie_id_tensor = torch.tensor([movie_id], dtype=torch.long, device=device)
        with torch.no_grad():
            rating = self(user_id_tensor, movie_id_tensor)
        return rating.item()

    def findUnseenMoviesByUser(self, target_user_idx):
        ratings = self.ratings
        movies = self.movies['movieId']
        ratingsUser = ratings.loc[(ratings['userId'] == target_user_idx)]
        seenMoviesList = ratingsUser[['movieId']].values.tolist()
        seenMovies = [item for sublist in seenMoviesList for item in sublist]
        unseenMovies = [movie for movie in movies if movie not in seenMovies]

        return unseenMovies

    def predictUnseenMoviesRating(self, userId):
        recommendations = []
        unseenMovies = self.findUnseenMoviesByUser(userId)
        for unseenMovie in unseenMovies:
            movieIdx = self.findIdx([unseenMovie])[0]
            rating = self.predictRatingMovie(userId-1, movieIdx)
            recommendations.append((unseenMovie, rating))
        recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)
        self.recommendations = recommendations
        return recommendations
    
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
        print(' Similarity with neuronal colaborative filter recommender: ' + str(sim))
        return sim

if __name__ == "__main__":

    # Load the dataset
    path_to_ml_latest_small = './ml-latest-small/'
    dataset = load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = split_users(dataset["ratings.csv"], val_movies)
    
    # Create matrix between user and movies 
    movies_idx = sorted(dataset["movies.csv"]["movieId"].tolist())
    users_idy = sorted(list(set(ratings_train["userId"].values)))
    
    start = time.time()
    target_user_idx = 1
    print('The prediction for user ' + str(target_user_idx) + ':')

    ncf = NeuronalColaborativeFilter(len(users_idy), len(movies_idx), ratings_train, dataset['movies.csv'])

    ## Hyper parameters
    lr = 1e-3   # Learning rate to update the model parameters
    wd = 1e-5   # Weight decay to avoid the overfitting

    # Batch size define how many data will be compute in an iteration, This help to improve the efficiency and accuracy
    batch_size = 64
    max_epochs = 50

    # The counter that avoid no reduction more than 3 epochs 
    early_stop_epoch_threshold = 5

    ncf.trainingModel(lr, wd, max_epochs, early_stop_epoch_threshold, batch_size)
    ncf.evaluateModel(ratings_train, batch_size)
    ncf.evaluateModel(ratings_val, batch_size)
    recommendations = ncf.predictUnseenMoviesRating(target_user_idx)
    ncf.printTopRecommendations()
    ncf.validation(ratings_val, target_user_idx)
    
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")
