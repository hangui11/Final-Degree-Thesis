import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

current_dir = os.path.dirname(os.path.abspath("__file__"))
code_development_dir = os.path.dirname(current_dir)
mf_dir = os.path.join(code_development_dir, "Experimentation/Matrix_factorization_experimentation")

'''
Load the different versions of the mf model from CSV files
'''
def read_csv(path):
    df = pd.read_csv(path)
    return df
    

if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(2049)

    mfPath = mf_dir + '/mfSim_v'
    
    mfSim1 = read_csv(mfPath+'1.csv')
    mf1 = mfSim1['mfSim'].values.tolist()

    mfSim2 = read_csv(mfPath+'2.csv')
    mf2 = mfSim2['mfSim'].values.tolist()

    mfSim3 = read_csv(mfPath+'3.csv')
    mf3 = mfSim3['mfSim'].values.tolist()

    mfSim4 = read_csv(mfPath+'4.csv')
    mf4 = mfSim4['mfSim'].values.tolist()

    mfSim5 = pd.read_csv(mfPath+'5.csv')
    mf5 = mfSim5['mfSim'].values.tolist()

    mfSim6 = pd.read_csv(mfPath+'6.csv')
    mf6 = mfSim6['mfSim'].values.tolist()

    # Get the number of users
    users = len(mfSim1)
    
     # For each version, count the number of users that have the highest similarity compared to the others versions
    mfCount1, mfCount2, mfCount3, mfCount4, mfCount5, mfCount6 = 0, 0, 0, 0, 0, 0, 
    for i in range(users):
        maxSim = max(mf1[i], mf2[i], mf3[i], mf4[i])
        if maxSim == mf1[i]: mfCount1 += 1
        if maxSim == mf2[i]: mfCount2 += 1
        if maxSim == mf3[i]: mfCount3 += 1
        if maxSim == mf4[i]: mfCount4 += 1
        if maxSim == mf5[i]: mfCount5 += 1
        if maxSim == mf6[i]: mfCount6 += 1

    version = ['config1', 'config2', 'config3', 'config4', 'config5', 'config6']
    versionSim = [mfCount1, mfCount2, mfCount3, mfCount4, mfCount5, mfCount6]
    plt.bar(version, versionSim)
    plt.xlabel('CONFIGURATIONS')
    plt.ylabel('NUMBER OF USERS')
    plt.title('NUMBER OF USERS WITH DIFFERENT CONFIGURATIONS')
    plt.savefig('../Images/number_users_mf.png')
    plt.show()


    ################################################################################################

    # Get the similarities for the random users
    n = 5
    X_axis = np.arange(n) 
    width = 0.6
    randomUsers = [np.random.randint(0, users) for i in range(n)]
    
    mfUsersSim1 = mfSim1.loc[mfSim1['userId'].isin(randomUsers)]['mfSim'].values.tolist()
    mfUsersSim2 = mfSim2.loc[mfSim2['userId'].isin(randomUsers)]['mfSim'].values.tolist()
    mfUsersSim3 = mfSim3.loc[mfSim3['userId'].isin(randomUsers)]['mfSim'].values.tolist()
    mfUsersSim4 = mfSim4.loc[mfSim4['userId'].isin(randomUsers)]['mfSim'].values.tolist()
    mfUsersSim5 = mfSim5.loc[mfSim5['userId'].isin(randomUsers)]['mfSim'].values.tolist()
    mfUsersSim6 = mfSim6.loc[mfSim6['userId'].isin(randomUsers)]['mfSim'].values.tolist()

    # Plot the similarities of different versions for the random users
    plt.figure(figsize=(12,8))
    plt.bar(X_axis-0.3, mfUsersSim1, width/6, label='config1')
    plt.bar(X_axis-0.2, mfUsersSim2, width/6, label='config2')
    plt.bar(X_axis-0.1, mfUsersSim3, width/6, label='config3')
    plt.bar(X_axis+0, mfUsersSim4, width/6, label='config4')
    plt.bar(X_axis+0.1, mfUsersSim5, width/6, label='config5')
    plt.bar(X_axis+0.2, mfUsersSim6, width/6, label='config6')
    plt.xticks(X_axis, randomUsers)
    plt.legend(loc='upper right', prop={"size": 8})
    plt.title('DIFFERENT SIMILARITY FOR RANDOM USERS')
    plt.xlabel('USERS ID')
    plt.ylabel('SIMILARITY')
    plt.savefig('../Images/mf_random_users.png', dpi=100)


