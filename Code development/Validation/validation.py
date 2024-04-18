import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    navPath = r"C:\Users\usuario\Desktop\FIB\Final-Degree-Thesis\Code development\Experimentation\Naive_experimentation\navSim.csv"
    navSim = pd.read_csv(navPath)
    nav = navSim['navSim'].sum()
    print(str(nav/len(navSim)))
    navUserList = navSim['userId'].values.tolist()
    navSimList = navSim['navSim'].values.tolist()
    plt.bar(navUserList, navSimList)
    plt.show()