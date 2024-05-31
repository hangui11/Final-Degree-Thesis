import sys
import os
import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))
code_development_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(code_development_dir)

from utils import *

'''
This class implements the DatasetBatchIterator class to iterate over the dataset in batches,
which shuffle the batches and return the batches in the form of tensors.
'''
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

'''
The class implements the NeuronalColaborativeFilter class, which is a neural network model that 
learns the latent factors of users and items based on their interactions in the dataset.
The model is trained using the Huber loss function and the Adam optimizer.
It's a subclass of nn.Module, which is a PyTorch's class that represents a neural network model.
'''
class NeuronalColaborativeFilter(nn.Module):
    def __init__(self, user_count, item_count, ratings_train, movies, embedding_size=64, hidden_layers=(128,64,32,16), dropout_rate=None, output_range=(1,5), k=5):
        super().__init__()
        self.movies = movies
        self.ratings = ratings_train
        # Split ratings train into x and y, where x is the user-item interaction matrix and y is the rating matrix
        self.ratings_x = ratings_train[['userId', 'movieId']]
        users = sorted(list(set(ratings_train["userId"].values)))
        self.ratings_x.loc[:,'userId'] = self.findIdxUser(self.ratings_x['userId'].values.tolist(), users)
        movies_idx = self.findIdx(self.ratings_x['movieId'].values.tolist())
        self.ratings_x.loc[:,'movieId'] = movies_idx
        self.ratings_y = ratings_train['rating'].astype(np.float32)
        
        # Store the number of the top recommendations to be generated
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
        
        # Initialize output normalization parameters
        assert output_range and len(output_range) == 2, "output range has to be tuple with two integers"
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[0]-output_range[1]) + 1
        
        # Initialitze the parameters for the model
        self.initParams()

    '''
    This function generates a MLP with the given parameters.
    '''
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
                # Dropout layer to reduce overfitting, if exist dropout_rate
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        # Conect the last layers with the output
        hidden_layers.append(nn.Linear(hidden_layers_units[-1], 1, device=self.device))
        # Reproduce output between 0 and 1 
        hidden_layers.append(nn.Sigmoid())
        
        return nn.Sequential(*hidden_layers)
    
    '''
    This function initializes the random parameters of the model, this parameters will be update
    in training process
    '''
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

        ## Initialize the value of embeddings with random number between -embedding_range and embedding_range
        embedding_range = 0.05
        self.user_embedding.weight.data.uniform_(-embedding_range, embedding_range)
        self.item_embedding.weight.data.uniform_(-embedding_range, embedding_range)

        ## Each layer of MLP will initialize according this function 'weights_init'
        self.MLP.apply(weights_init)

    '''
    This function compute the forward pass of the model, which is the computation of the output of the model
    given the input user_id and item_id
    The function will be auto invoked by PyTorch when we call the model.
    '''
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

    '''
    This function find the index of the movieId in the movies list
    '''
    def findIdx(self, movie_ids):
        idx = []
        listMovies = self.movies['movieId'].values.tolist()
        for movie_id in movie_ids:
            if movie_id in listMovies:
                idx.append(listMovies.index(movie_id))
            else:
                print("Error: movie_id not found")
                break
        return idx
    
    '''
    This method fint the index of the userId in the users list
    '''
    def findIdxUser(self, user_ids, users):
        idx = []
        for user_id in user_ids:
            if user_id in users:
                idx.append(users.index(user_id))
            else:
                print("Error: user_id not found")
                break
        return idx

    
    '''
    The training function of the model, which will update the parameters of the model using
    the Huber loss function and the Adam optimizer
    '''
    def trainingModel(self, lr, wd, max_epochs, early_stop_epoch_threshold, batch_size):
        # self.initParams()
        self.train()

        # Training loop control parameters
        no_loss_reduction_epoch_counter = 0
        min_loss = np.inf
        
        # Use GPU to run the Neuronal Network
        self.to(self.device)

        # Loss function: Huber Error -> combination of MAE and MSE
        beta = 0.25
        loss_criterion = nn.SmoothL1Loss(reduction='sum', beta=beta)
        # Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        # Training of the model
        print('Start the NCF MODEL training .....')
        
        # Each epoch update the parameters of the model
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            # Iterate over the batches of the dataset
            for x_batch, y_batch in DatasetBatchIterator(self.ratings_x, self.ratings_y, batch_size=batch_size, shuffle=True):
                x_batch, y_batch = x_batch.to(device=self.device), y_batch.to(device=self.device)

                # Zero the gradients of the model's parameters
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self(x_batch[:, 0], x_batch[:, 1])
                    # Compute the loss of the model's output with respect to the true labels
                    loss = loss_criterion(outputs, y_batch)
                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # Update the parameters of the model using the gradients
                    optimizer.step()
                epoch_loss += loss.item()
        
            epoch_loss = epoch_loss / len(self.ratings_x)
            # print(f'Epoch: {epoch+1}, Loss: {epoch_loss}')

            # Check the early stop condition of the model
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                no_loss_reduction_epoch_counter = 0
            else:
                no_loss_reduction_epoch_counter += 1
            if no_loss_reduction_epoch_counter >= early_stop_epoch_threshold:
                print(f'Early stop at epoch {epoch+1}')
                break

    '''
    The evaluation function of the model, which will compute the RMSE error of the model
    given the validation dataset
    '''
    def evaluateModel(self, ratings_val, batch_size, users):
        self.eval()
        # Split ratings validation dataset into x and y, where x is the user-item interaction matrix and y is the rating matrix
        ratings_val_x = ratings_val[['userId', 'movieId']]
        ratings_val_x.loc[:,'userId'] = self.findIdxUser(ratings_val_x['userId'].values.tolist(), users)
        movies_idx = self.findIdx(ratings_val_x['movieId'].values.tolist())
        ratings_val_x.loc[:,'movieId'] = movies_idx
        ratings_val_y = ratings_val[['rating']]
        groud_truth, predictions = [], []

        with torch.no_grad():
            for x_batch, y_batch in DatasetBatchIterator(ratings_val_x, ratings_val_y, batch_size=batch_size, shuffle=False):
                x_batch, y_batch = x_batch.to(device=self.device), y_batch.to(device=self.device)
                # Get predictions from the model
                outputs = self(x_batch[:, 0], x_batch[:, 1])
                groud_truth.extend(y_batch.tolist())
                predictions.extend(outputs.tolist())
        groud_truth = np.array(groud_truth).ravel()
        predictions = np.array(predictions).ravel()
        # Using RMSE algorithm to evaluate the model's accuracy
        RMSE = np.sqrt(mean_squared_error(groud_truth, predictions))
        print(f'RMSE: {RMSE}')

    '''
    The prediction function of the model, which will predict the rating of a movie for an user
    '''
    def predictRatingMovie(self, user_id, movie_id):
        self.eval()
        device = next(self.parameters()).device
        user_id_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        movie_id_tensor = torch.tensor([movie_id], dtype=torch.long, device=device)
        with torch.no_grad():
            rating = self(user_id_tensor, movie_id_tensor)
        return rating.item()

    '''
    The function that find the unseen movies of an user
    '''
    def findUnseenMoviesByUser(self, target_user_idx):
        ratings = self.ratings
        movies = self.movies['movieId']
        ratingsUser = ratings.loc[(ratings['userId'] == target_user_idx)]
        seenMoviesList = ratingsUser[['movieId']].values.tolist()
        seenMovies = [item for sublist in seenMoviesList for item in sublist]
        unseenMovies = [movie for movie in movies if movie not in seenMovies]
        return unseenMovies

    '''
    The function that predit the recommendations of unseen movies for an user
    '''
    def predictUnseenMoviesRating(self, userId, users):
        recommendations = []
        unseenMovies = self.findUnseenMoviesByUser(userId)
        for unseenMovie in unseenMovies:
            movieIdx = self.findIdx([unseenMovie])[0]
            userIdx = self.findIdxUser([userId], users)[0]
            rating = self.predictRatingMovie(userIdx, movieIdx)
            recommendations.append((unseenMovie, rating))
        # Sort the recommendations by rating in descending order
        recommendations = sorted(recommendations, key=lambda x:x[1], reverse=True)
        self.recommendations = recommendations
        return recommendations
    
    '''
    This function prints the top recommendations generated by the Neuronal Collaborative Filtering model
    '''
    def printTopRecommendations(self):
        for recomendation in self.recommendations[:self.topK]:
            rec_movie = self.movies[self.movies["movieId"]  == recomendation[0]]
            print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    '''
    Method to compute the similarity between predictions and the validation dataset,
    which is the same as the similarity between the validation movies genres and the recommended movies genres
    '''
    def validation(self, ratings_val, target_user_idx):
        # Validation
        matrixmpa_genres, validationMoviesGenress = validationMoviesGenres(self.movies, ratings_val, target_user_idx)

        topMoviesUser = list(list(zip(*self.recommendations[:self.topK]))[0])
        recommendsMoviesUser = matrixmpa_genres.loc[topMoviesUser]
        
        # Compute the similarity between the validation movies genres and the recommended movies genres
        sim = cosinuSimilarity(validationMoviesGenress, recommendsMoviesUser)
        # print(' Similarity with neuronal colaborative filter recommender: ' + str(sim))
        return sim

if __name__ == "__main__":
    # Set the random seed for reproducibility
    number = 9101307
    torch.manual_seed(number)
    np.random.seed(number)
    
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
    wd = 1e-4   # Weight decay to avoid the overfitting

    # Batch size define how many data will be compute in an iteration, This help to improve the efficiency and accuracy
    batch_size = 64
    max_epochs = 50

    # The counter that avoid no reduction more than 3 epochs 
    early_stop_epoch_threshold = 5

    ncf.trainingModel(lr, wd, max_epochs, early_stop_epoch_threshold, batch_size)
    # ncf.evaluateModel(ratings_train, batch_size)
    ncf.evaluateModel(ratings_val, batch_size, users_idy)
    recommendations = ncf.predictUnseenMoviesRating(target_user_idx)
    ncf.printTopRecommendations()
    ncf.validation(ratings_val, target_user_idx)
    
    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")
