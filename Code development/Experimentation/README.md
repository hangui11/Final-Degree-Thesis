## Experimentation

In this section, we will experiment different models using three dataset, two of them are obtained from the original dataset, which are reduced versions. We will experiment the models for all users in the datasets and store the results in CSV files.

The following files contains the experimetion scripts and the CSV files with the results:

- **Item_experimentation**: The file contains the experimentation for the Item-based collaborative filtering model and the results are stored in the CSV files.

- **kNN_hybrid_experimentation**: The file contains the experimentation for the kNN-based hybrid model and the results are stored in the CSV files.

- **Matrix_factorization_experimentation**: The file contains the experiments for the matrix factorization model and the results are stored in the CSV files. Also contains the results of differents configurations of the matrix factorization model.

- **Neuronal_colaborative_filter_experimentation**: The file contains the experiments for the neural collaborative filtering model and the results are stored in the CSV files. Also contains the RMSE accuracy results and thre results of differents configurations of the neural collaborative filtering model.

- **Trivial_experimentation**: The file contains the experiments for the trivial model and the results are stored in the CSV files.

- **User_experimentation**: The file contains the experimentation for the User-based collaborative filtering model and the results are stored in the CSV files.

- **models_execution_time.txt**: The file contains the execution time of the models for all users in the datasets "ratings.csv".

- **pseudocodeValidation.txt**: The file contains the pseudocode of the validation process for the models.

To execute the script, it's necessary get into the directory of the file and run the following command:

```
python script_name.py
```

For example, to execute the Trivial_experimentation, run the following command:

```
python3 trivial_recommender_experimentation.py
```