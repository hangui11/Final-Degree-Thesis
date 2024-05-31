import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath("__file__"))
code_development_dir = os.path.dirname(current_dir)
ncf_dir = os.path.join(code_development_dir, "Experimentation/Neuronal_colaborative_filter_experimentation")

'''
Load the different versions of the ncf model from CSV files
'''
def read_csv(path):
    df = pd.read_csv(path)
    return df
    
if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(8)

    # Load the different versions of the ncf model
    ncfPath = ncf_dir + '/ncfSim_v'

    ncfSim1 = pd.read_csv(ncfPath+'1.csv')
    ncf1 = ncfSim1['ncfSim'].values.tolist()

    ncfSim2 = pd.read_csv(ncfPath+'2.csv')
    ncf2 = ncfSim2['ncfSim'].values.tolist()

    ncfSim3 = pd.read_csv(ncfPath+'3.csv')
    ncf3 = ncfSim3['ncfSim'].values.tolist()

    ncfSim4 = pd.read_csv(ncfPath+'4.csv')
    ncf4 = ncfSim4['ncfSim'].values.tolist()

    ncfSim5 = pd.read_csv(ncfPath+'5.csv')
    ncf5 = ncfSim5['ncfSim'].values.tolist()

    ncfSim6 = pd.read_csv(ncfPath+'6.csv')
    ncf6 = ncfSim6['ncfSim'].values.tolist()

    # Get the number of users
    users = len(ncfSim1)

    # For each version, count the number of users that have the highest similarity compared to the others versions
    ncfCount1, ncfCount2, ncfCount3, ncfCount4, ncfCount5, ncfCount6 = 0, 0, 0, 0, 0, 0
    for i in range(users):
        maxSim = max(ncf1[i], ncf2[i], ncf3[i], ncf4[i], ncf5[i], ncf6[i])
        if (ncf1[i] == maxSim): ncfCount1 += 1
        if (ncf2[i] == maxSim): ncfCount2 += 1
        if (ncf3[i] == maxSim): ncfCount3 += 1
        if (ncf4[i] == maxSim): ncfCount4 += 1
        if (ncf5[i] == maxSim): ncfCount5 += 1
        if (ncf6[i] == maxSim): ncfCount6 += 1
    version = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    versionSim = [ncfCount1, ncfCount2, ncfCount3, ncfCount4, ncfCount5, ncfCount6]
    plt.bar(version, versionSim)
    plt.xlabel('VERSIONS')
    plt.ylabel('NUMBER OF USERS')
    plt.title('NUMBER OF USERS WITH DIFFERENT VERSIONS')
    plt.savefig('../Images/number_users_ncf.png')
    plt.show()


    ########################################################################

    # Get the similarities for the random users
    n = 5
    X_axis = np.arange(n) 
    width = 0.7
    randomUsers = [np.random.randint(0, users) for i in range(n)]
    
    ncfUsersSim1 = ncfSim1.loc[ncfSim1['userId'].isin(randomUsers)]['ncfSim'].values.tolist()
    ncfUsersSim2 = ncfSim2.loc[ncfSim2['userId'].isin(randomUsers)]['ncfSim'].values.tolist()
    ncfUsersSim3 = ncfSim3.loc[ncfSim3['userId'].isin(randomUsers)]['ncfSim'].values.tolist()
    ncfUsersSim4 = ncfSim4.loc[ncfSim4['userId'].isin(randomUsers)]['ncfSim'].values.tolist()
    ncfUsersSim5 = ncfSim5.loc[ncfSim5['userId'].isin(randomUsers)]['ncfSim'].values.tolist()
    ncfUsersSim6 = ncfSim6.loc[ncfSim6['userId'].isin(randomUsers)]['ncfSim'].values.tolist()

    # Plot the similarities of different versions for the random users
    plt.figure(figsize=(12,8))
    plt.bar(X_axis-0.3, ncfUsersSim1, width/7, label='v1')
    plt.bar(X_axis-0.2, ncfUsersSim2, width/7, label='v2')
    plt.bar(X_axis-0.1, ncfUsersSim3, width/7, label='v3')
    plt.bar(X_axis+0, ncfUsersSim4, width/7, label='v4')
    plt.bar(X_axis+0.1, ncfUsersSim5, width/7, label='v5')
    plt.bar(X_axis+0.2, ncfUsersSim6, width/7, label='v6')
    plt.xticks(X_axis, randomUsers)
    plt.legend(loc='upper right', prop={"size": 10})
    plt.title('DIFFERENT SIMILARITY FOR RANDOM USERS')
    plt.xlabel('VERSION OF NCF')
    plt.ylabel('SIMILARITY')
    plt.savefig('../Images/ncf_random_users.png', dpi=100)

    # Plot the accuracy for each version, which the accuracies is obtained from RMSE algorithm
    accuracyPath = ncf_dir + "/versionAccuracy.csv"
    accuracy = pd.read_csv(accuracyPath)
    accuracy = accuracy['accuracy'].values.tolist()
    plt.figure()
    plt.cla()
    plt.bar(version, accuracy)
    plt.xlabel('VERSIONS')
    plt.ylabel('ACCURACY')
    plt.title('ACCURACY FOR DIFFERENT VERSIONS')
    
    plt.savefig('../Images/accuracy_ncf.png')
    plt.show()
