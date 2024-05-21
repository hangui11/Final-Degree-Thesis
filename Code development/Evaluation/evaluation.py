import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

current_dir = os.path.dirname(os.path.abspath("__file__"))
code_development_dir = os.path.dirname(current_dir)
experimentation_dir = os.path.join(code_development_dir, "Experimentation")

if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(123456)

    # Load the dataset
    trivialPath = experimentation_dir + "/Trivial_experimentation/trivialSim.csv"
    trivialSim = pd.read_csv(trivialPath)
    trivial = trivialSim['trivialSim'].values.tolist()

    userPath = experimentation_dir + "/User_experimentation/userSim.csv"
    userSim = pd.read_csv(userPath)
    user = userSim['userSim'].values.tolist()
    
    itemPath = experimentation_dir + "/Item_experimentation/itemSim.csv"
    itemSim = pd.read_csv(itemPath)
    item = itemSim['itemSim'].values.tolist()

    knnPath = experimentation_dir + "/kNN_hybrid_experimentation/knnSim.csv"
    knnSim = pd.read_csv(knnPath)
    knn = knnSim['knnSim'].values.tolist()

    mfPath = experimentation_dir + "/Matrix_factorization_experimentation/mfSim.csv"
    mfSim = pd.read_csv(mfPath)
    mf = mfSim['mfSim'].values.tolist()

    ncfPath = experimentation_dir + "/Neuronal_colaborative_filter_experimentation/ncfSim.csv"
    ncfSim = pd.read_csv(ncfPath)
    ncf = ncfSim['ncfSim'].values.tolist()

    users = len(trivialSim)
    n = 5
    X_axis = np.arange(n) 
    width = 0.6
    randomUsers = [np.random.randint(0, users) for i in range(n)]
    
    trivialUsersSim = trivialSim.loc[trivialSim['userId'].isin(randomUsers)]['trivialSim'].values.tolist()
    userUsersSim = userSim.loc[userSim['userId'].isin(randomUsers)]['userSim'].values.tolist()
    itemUsersSim = itemSim.loc[itemSim['userId'].isin(randomUsers)]['itemSim'].values.tolist()
    knnUsersSim = knnSim.loc[knnSim['userId'].isin(randomUsers)]['knnSim'].values.tolist()
    mfUsersSim = mfSim.loc[mfSim['userId'].isin(randomUsers)]['mfSim'].values.tolist()
    ncfUsersSim = ncfSim.loc[ncfSim['userId'].isin(randomUsers)]['ncfSim'].values.tolist()

    plt.figure(figsize=(12,8))
    plt.bar(X_axis-0.3, trivialUsersSim, width/6, label='trivial')
    plt.bar(X_axis-0.2, userUsersSim, width/6, label='user')
    plt.bar(X_axis-0.1, itemUsersSim, width/6, label='item')
    plt.bar(X_axis+0, knnUsersSim, width/6, label='knn')
    plt.bar(X_axis+0.1, mfUsersSim, width/6, label='mf')
    plt.bar(X_axis+0.2, ncfUsersSim, width/6, label='ncf')
    plt.xticks(X_axis, randomUsers)
    plt.legend(loc='upper right', prop={"size": 10})
    plt.title('DIFFERENT SIMILARITY FOR RANDOM USERS')
    plt.xlabel('USER ID')
    plt.ylabel('SIMILARITY')
    plt.savefig('../Images/random_users.png', dpi=100)
    
    ############################################################################

    trivialCount = 0
    userCount = 0
    itemCount = 0
    knnCount = 0
    mfCount = 0
    ncfCount = 0
    
    for i in range(users):
        maxSim = max(trivial[i], user[i], item[i], knn[i], mf[i], ncf[i])

        if (maxSim == trivial[i]): trivialCount += 1
        if (maxSim == user[i]): userCount += 1
        if (maxSim == item[i]): itemCount += 1
        if (maxSim == knn[i]): knnCount += 1
        if (maxSim == mf[i]): mfCount += 1
        if (maxSim == ncf[i]): ncfCount += 1

    methods = ['trivial', 'user', 'item', 'knn', 'mf', 'ncf']
    methodsSim = [trivialCount, userCount, itemCount, knnCount, mfCount, ncfCount]
    color = ['red', 'green', 'red', 'blue', 'blue', 'blue']
    
    plt.figure(figsize=(12,8))
    plt.bar(methods, methodsSim, color=color)

    # Method 1: Line2D
    # legend_elements = [Line2D([0], [0], color='red', lw=4, label='Item-based model'),
    #                Line2D([0], [0], color='green', lw=4, label='User-based model'),
    #                Line2D([0], [0], color='blue', lw=4, label='Hybrid model')]
    # plt.legend(handles=legend_elements, loc='upper right')

    # Method 2: Rectangle
    colors = {'Item-based model':'red', 'User-based model': 'green', 'Hybrid model': 'blue'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0), 1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, loc='upper right', prop={"size": 10})

    plt.xlabel('METHODS')
    plt.ylabel('USERS COUNTER')
    plt.title('COMPARISION OF ACCURACY FOR DIFFERENT METHODS')
    plt.savefig('../Images/number_users.png')
    # print(methodsSim)
    