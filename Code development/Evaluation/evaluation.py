import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":

    np.random.seed(123456)
    navPath = r"C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Naive_experimentation\navSim1.csv"
    navSim = pd.read_csv(navPath)
    nav = navSim['navSim'].values.tolist()
    
    userPath = r"C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\User_experimentation\userSim1.csv"
    userSim = pd.read_csv(userPath)
    user = userSim['userSim'].values.tolist()
    
    itemPath = r"C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Item_experimentation\itemSim1.csv"
    itemSim = pd.read_csv(itemPath)
    item = itemSim['itemSim'].values.tolist()

    knnPath = r"C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\kNN_hybrid_experimentation\knnSim1.csv"
    knnSim = pd.read_csv(knnPath)
    knn = knnSim['knnSim'].values.tolist()

    mfPath = r"C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Matrix_factorization_experimentation\mfSim1.csv"
    mfSim = pd.read_csv(mfPath)
    mf = mfSim['mfSim'].values.tolist()

    ncfPath = r"C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Neuronal_colaborative_filter_experimentation\ncfSim2.csv"
    ncfSim = pd.read_csv(ncfPath)
    ncf = ncfSim['ncfSim'].values.tolist()

    users = len(navSim)
    X_axis = np.arange(5) 
    width = 0.6
    randomUsers = [np.random.randint(0, users) for i in range(5)]
    
    navUsersSim = navSim.loc[navSim['userId'].isin(randomUsers)]['navSim'].values.tolist()
    userUsersSim = userSim.loc[userSim['userId'].isin(randomUsers)]['userSim'].values.tolist()
    itemUsersSim = itemSim.loc[itemSim['userId'].isin(randomUsers)]['itemSim'].values.tolist()
    knnUsersSim = knnSim.loc[knnSim['userId'].isin(randomUsers)]['knnSim'].values.tolist()
    mfUsersSim = mfSim.loc[mfSim['userId'].isin(randomUsers)]['mfSim'].values.tolist()
    ncfUsersSim = ncfSim.loc[ncfSim['userId'].isin(randomUsers)]['ncfSim'].values.tolist()

    plt.bar(X_axis-0.3, navUsersSim, width/6, label='nav')
    plt.bar(X_axis-0.2, userUsersSim, width/6, label='user')
    plt.bar(X_axis-0.1, itemUsersSim, width/6, label='item')
    plt.bar(X_axis+0, knnUsersSim, width/6, label='knn')
    plt.bar(X_axis+0.1, mfUsersSim, width/6, label='mf')
    plt.bar(X_axis+0.2, ncfUsersSim, width/6, label='ncf')
    plt.xticks(X_axis, randomUsers)
    plt.legend()
    plt.title('DIFFERENT SIMILARITY FOR RANDOM USERS')
    plt.xlabel('USER ID')
    plt.ylabel('SIMILARITY')
    plt.show()

    navCount = 0
    userCount = 0
    itemCount = 0
    knnCount = 0
    mfCount = 0
    ncfCount = 0
    
    for i in range(users):
        maxSim = max(nav[i], user[i], item[i], knn[i], mf[i], ncf[i])

        if (maxSim == nav[i]): navCount += 1
        if (maxSim == user[i]): userCount += 1
        if (maxSim == item[i]): itemCount += 1
        if (maxSim == knn[i]): knnCount += 1
        if (maxSim == mf[i]): mfCount += 1
        if (maxSim == ncf[i]): ncfCount += 1

    methods = ['nav', 'user', 'item', 'knn', 'mf', 'ncf']
    methodsSim = [navCount, userCount, itemCount, knnCount, mfCount, ncfCount]
    plt.bar(methods, methodsSim)
    plt.xlabel('METHODS')
    plt.ylabel('USERS COUNTER')
    plt.title('COMPARISION OF ACCURACY FOR DIFFERENT METHODS')
    plt.show()
    