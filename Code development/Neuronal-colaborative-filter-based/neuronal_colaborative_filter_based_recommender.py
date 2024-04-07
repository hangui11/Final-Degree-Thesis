import torch
import utils as ut
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim

class NeuronalColaborativeFilter(nn.Module):

    def __init__(self, user_count, item_count, embedding_size=64, hidden_layers=(128,64,32,16,8), dropout_rate=0.1, output_range=(1,5)):
        super().__init__()

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
        self.norm_range = abs(output_range[0] - output_range[1]) + 1

        self.initParams()

    def __genMLP(self, embedding_size, hidden_layers_units, dropout_rate):
        assert (embedding_size * 2) == hidden_layers_units[0], "First input layers number units must be the twice time of embedding size"

        hidden_layers = []
        input_units = hidden_layers_units[0]

        for num_units in hidden_layers_units[1:]:
            # Connect the inputs_units layers with num_units layers linealy
            hidden_layers.append(nn.Linear(input_units, num_units))
            hidden_layers.append(nn.BatchNorm1d(num_units))
            # Learn the relation between no connect layers
            hidden_layers.append(nn.ReLU())

            if dropout_rate:
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        # Conect with the output
        hidden_layers.append(nn.Linear(hidden_layers_units[-1], 1))
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
        embedding_range = 0.1
        self.user_embedding.weight.data.uniform_(-embedding_range, embedding_range)
        self.item_embedding.weight.data.uniform_(-embedding_range, embedding_range)

        ## Each layer of MLP will initialize according this function 'weights_init'
        self.MLP.apply(weights_init)

    ## This function will be auto invoked
    def forward(self, user_id, item_id):
        ## Access the features of user_id and item_id
        user_features = self.user_embedding(user_id % self.user_hash_size)
        item_features = self.item_embedding(item_id % self.user_hash_size)

        ## Concat the features of user and item in one feature representation
        x = torch.cat([user_features, item_features], dim=1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        ## The feature will be compute with neuronal network 
        x = self.MLP(x)

        ## Normalize the output between 1 and 5
        normalized_output = x * self.norm_range
        return normalized_output

    def trainingModel(self, lr, wd, max_epochs, early_stop_epoch_threshold, batch_size, ratings_train):
        # self.initParams()
        self.train(True)
        ## Training loop control parameters
        no_loss_reduction_epoch_counter = 0
        min_loss = np.inf
        min_loss_model_weights = None
        
        # Use GPU to run the Neuronal Network
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        ## Loss function: Huber Error -> combination of MAE and MSE
        beta = 0.5
        loss_criterion = nn.SmoothL1Loss(reduction='sum', beta=beta)
        ## Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        ## Training of the model
        print('Start training.....')
        
        ## Each epoch update our model
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            
            # Shuffle the training data to help the learning to avoid the overfitting and improve the model generalization
            shuffled_indices = np.random.permutation(len(ratings_train))
            shuffled_ratings_train = ratings_train.iloc[shuffled_indices]
            
            # Iterate over batches
            for i in range(0, len(ratings_train), batch_size):
                batch = shuffled_ratings_train[i:i+batch_size]
                user_ids = torch.tensor(batch['userId'].values, dtype=torch.long, device=device)
                item_ids = torch.tensor(batch['movieId'].values, dtype=torch.long, device=device)
                ratings = torch.tensor(batch['rating'].values, dtype=torch.float, device=device)
                
                # Initialize the gradients
                optimizer.zero_grad()

                # Compute the predictions
                outputs = self(user_ids, item_ids)

                # Compute the loss function
                loss = loss_criterion(outputs.squeeze(), ratings)
                loss.backward()

                # Update the parameters of NCF
                optimizer.step()

                # Acumulate the total error
                epoch_loss += loss.item()

            # Mean epoch loss
            epoch_loss /= len(ratings_train)
            # print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_loss}')

            # Early stopping based on validation loss
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                min_loss_model_weights = self.state_dict()
                no_loss_reduction_epoch_counter = 0
            else:
                no_loss_reduction_epoch_counter += 1

            if no_loss_reduction_epoch_counter >= early_stop_epoch_threshold:
                print(f'Early stopping at epoch {epoch+1} due to no reduction in loss for {early_stop_epoch_threshold} epochs.')
                break

        # Load the model weights with the minimum validation loss
        if min_loss_model_weights:
            self.load_state_dict(min_loss_model_weights)

    

## Prediction of rating for a movie
def predictRatingMovie(ncf_model, user_id, movie_id):
    ncf_model.eval()
    device = next(ncf_model.parameters()).device
    user_id_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
    movie_id_tensor = torch.tensor([movie_id], dtype=torch.long, device=device)
    with torch.no_grad():
        rating = ncf_model(user_id_tensor, movie_id_tensor)
    return rating.item()

def findUnseenMoviesByUser(ratings, target_user_idx, movies):
    ratingsUser = ratings.loc[(ratings['userId'] == target_user_idx)]
    seenMoviesList = ratingsUser[['movieId']].values.tolist()
    seenMovies = [item for sublist in seenMoviesList for item in sublist]
    unseenMovies = [movie for movie in movies if movie not in seenMovies]

    return unseenMovies

def predictUnseenMoviesRating(ncf, unseenMovies, userId):
    recommendations = []
    for movieId in unseenMovies:
        rating = predictRatingMovie(ncf, userId, movieId)
        recommendations.append((movieId, rating))

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
    movies_idx = sorted(dataset["movies.csv"]["movieId"].tolist())
    users_idy = sorted(list(set(ratings_train["userId"].values)))
    
    start = time.time()
    target_user_idx = 430
    print('The prediction for user ' + str(target_user_idx) + ':')

    unseenMovies = findUnseenMoviesByUser(ratings_train, target_user_idx, movies_idx)

    ncf = NeuronalColaborativeFilter(len(users_idy), len(movies_idx))

    ## Hyper parameters
    lr = 1e-3   # Learning rate
    wd = 1e-4   # Weight decay to avoid the overfitting

    # Batch size define how many data will be compute in an iteration, This help to improve the efficiency and accuracy
    batch_size = 1024
    max_epochs = 500

    # The counter that avoid no reduction more than 3 epochs 
    early_stop_epoch_threshold = 5

    ncf.trainingModel(lr, wd, max_epochs, early_stop_epoch_threshold, batch_size, ratings_train)

    recommendations = predictUnseenMoviesRating(ncf, unseenMovies, target_user_idx)

    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation: Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    # Validation
    matrixmpa_genres, validationMoviesGenres = ut.validationMoviesGenres(dataset["movies.csv"], ratings_val, target_user_idx)

    # neuronal colaborative filter based recommender
    topMoviesUser = list(list(zip(*recommendations[:5]))[0])
    recommendsMoviesGenresItem = matrixmpa_genres.loc[topMoviesUser]
    # sim entre matriu genere amb recomanador neuronal colaborative filter
    sim = ut.cosinuSimilarity(validationMoviesGenres, recommendsMoviesGenresItem)
    print(' Similarity with neuronal colaborative filter recommender: ' + str(sim))

    end = time.time()
    print("The execution time: " + str(end-start) + " seconds")