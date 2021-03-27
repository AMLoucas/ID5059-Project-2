# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys as sys

# for fitting a decision tree
from sklearn.tree import DecisionTreeRegressor

# this import an renaming is needed to import missforest
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
# Importing the library from python to impute the missing values with MissForest tactic.
from missingpy import MissForest
# Import the KNNImputer
from sklearn.impute import KNNImputer
# Import simpleimputer to import with median
from sklearn.impute import SimpleImputer

# Can create a confusion matrix with accuracy to see how good model is.
from sklearn import metrics
# Importing accuracy_score to auto calculate our classifiers accuracy
from sklearn.metrics import accuracy_score

# Importing so we can create confusion matrix images.
import matplotlib.pyplot as plt
import seaborn as sn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Set seed for reproducible reasons. Achieving same modified set every run.
# todo: check if the same seed works from python to R, hence avoid storing data and just running same analysis on R.

import random
import time
# random.seed(5059)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### IMPORT DATA

original_df = pd.read_csv("data/train.csv")
original_df.head(6)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### FUNCTION TO REMOVE DATA

def remove_random_values(df, columns, percentage):
    """
    Funtion the will remove values from the selected columns at random,
    then replace the removed values with np.nan which is NaN.

    PSEUDOCODE:

    1. Random sample row indexes (with replacement)
    2. Random sample column labels (with replacement)
    3. Extract random entry and replace with np.nan
    4. Repeat (1-3) until total proportion of data has been extracted

    :param df: pandas DataFrame
    :param columns: list of columns to 'drop' values from. Note: must be string name of columns
    :param percentage: float input such as 0.20 (i.e. 20%)
    :return: DataFrame with NaN entries
    """

    start = time.time()
    # Set up: Focus on subset of data dependent on columns,
    #         hence extract information about subset and number of values to drop
    # df = df.copy()
    subset = df[columns].copy()
    nrow, ncol = subset.shape
    n_samples = int(percentage*nrow)

    # 1. Pick out a vector of random samples (row indexes)
    row_indexes = np.arange(0, nrow, 1).tolist()

    # 2. Sample with replacement from the list of possible indexes
    #    note, `choices` is used to sample with replacement
    sampled_row_indexes = random.choices(row_indexes, k = n_samples)
    sampled_columns = random.choices(columns, k = n_samples)

    # true_value = []
    # 3. Replace all entries with NaN
    for i in range(0, n_samples):
        # true_value.append([subset.loc[sampled_row_indexes[i], sampled_columns[i]], sampled_row_indexes[i], sampled_columns[i]])
        subset.loc[sampled_row_indexes[i], sampled_columns[i]] = 'nan'

    # note having it as a loop ensures we are not holding a massive dataframe in memory so its a lot faster.
    # e.g. the vectorised version subset.loc[sampled_row_indexes, sampled_columns] = np.nan is too expensive

    # 4. Update data frame with new NaN values
    df.update(subset)
    df = df.replace('nan', np.nan)

    end = time.time()
    print("Time Elapsed : ", end - start)

    return(df)

### Sample runs using the entire dataset:

# discrete only case
df = remove_random_values(original_df.copy(), ['cat0', 'cat1'], 0.2)
# Time Elapsed : 4.55 seconds

# continuous only case
df = remove_random_values(original_df.copy(), ['cont0', 'cont1'], 0.2)
# Time Elapsed : 250.5184588432312/60 = 4 minutes ??

# both discrete and continuous cases
# df = remove_random_values(original_df, ['cat1', 'cont2'], 0.2)
# Time Elapsed : 96.34 seconds ( why so long? )
# Bizzare

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### FUNCTION TO RUN MODELS

# todo: this only currently works for continuous features - none of the methods support categorical features
#       need to look into setting up a pipeline which supports both.
#       Look into `FeatureUnion` it is sort of like a Pipeline thing but suitable for imputations.

def imputation_methods(df, complete_df, columns):
    """
    Function which runs through our chosen three methods: Median Imputation, MissForest and KNNImputer.

    :param df: dataframe which has missing data - subset on the columns to impute only
    :param complete_df: dataframe which has the complete data
    :param columns: list of columns strings used in the remove_random_values() function.
    :return: the three imputed datasets (for now)
    """

    start = time.time()

    ##### DEPLOY METHODS:

    ### STRATEGY 1: MEDIAN IMPUTATION

    # Impute Data
    median_imputation = df.copy()
    median_imputer = SimpleImputer(strategy = "median")
    median_imputer.fit(median_imputation)

    # Once the data has been impute is outputs in an array
    RawOutput = median_imputer.transform(median_imputation)

    # We want to convert the array back to data frame now for further computations.
    median_imputation_data = pd.DataFrame(RawOutput,
                                     columns = median_imputation.columns,
                                     index = median_imputation.index)

    del median_imputation
    del RawOutput

    middle1 = time.time()

    print("Median Imputation Complete - Time Elapsed :", middle1 - start)

    ### STRATEGY 2: MissForest

    miss_forest_data = df.copy()

    # Create imputation tactic.
    miss_forest_imputer = MissForest()
    # Once the data has been impute is outputs in an array
    RawOutput = miss_forest_imputer.fit_transform(miss_forest_data)

    # We want to convert the array back to data frame now for further computations.
    miss_forest_data = pd.DataFrame(RawOutput,
                                     columns = miss_forest_data.columns,
                                     index = miss_forest_data.index)

    middle2 = time.time()

    print("MissForest Imputation Complete - Time Elapsed :", middle2 - start)

    ### STRATEGY 3: KNNImputer

    knn_imputer_data = df.copy()

    # We can adjust hyperparameter n_neighbors.
    knn_imputer = KNNImputer(n_neighbors = 2, weights = "uniform")

    # Imputing values
    RawOutput = knn_imputer.fit_transform(knn_imputer_data)

    # We want to convert the array back to data frame now for further computations.
    knn_imputation_data = pd.DataFrame(RawOutput,
                                     columns = knn_imputer_data.columns,
                                     index = knn_imputer_data.index)

    middle3 = time.time()

    del knn_imputer_data
    del RawOutput

    print("KNN Imputation Complete - Time Elapsed :", middle3 - middle2)

    ##### FOCUS ON PERFORMANCE METRICS:

    ### EXTRACT A LIST OF ORIGINAL DATA WHICH BECAME NaN ENTRIES

    # todo: finish function including evaluating performance metrics for each list of imputed data vs true data

    """
    
    # true_values = []
    # true_values_boolean = []
    for column in columns:

        # 1. Extract boolean vector (True/False) to identify Imputed True Data Vectors
        boolean_values = (complete_df != df)[column].to_numpy().tolist()
        values = original_df[column][boolean_values]
        # print("for column", column, "we have :", len(values), "true values which were dropped")

        # 2. Store away true value and boolean vector
        # true_values.append(values)
        # true_values_boolean.append(boolean_values)

        # 3. Extract Imputed Data Vectors
        values_median_imputation = median_imputation_data[column][boolean_values]
        values_knn_imputation = knn_imputation_data[column][boolean_values]

        # 4. Evaluate performance metrics 
        # need to do some adjusting of the input of the function.

    end = time.time()

    print("Performance Metrics Computed - Time Elapsed :", end - middle2)
    
    """

    return(median_imputation_data, miss_forest_data, knn_imputation_data)

results = imputation_methods(df[['cont0', 'cont1']], None, None)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### PERFORMANCE METRICS FUNCTION

def performance_metrics(imputed_values, true_values):
    """
    Function to calculate all performance metrics of interest for each list of imputed vs true values.

    VERY IMPORTANT: **each list for now is specific to each column**.
    todo: add a layer to this function for which you can input a list of lists (i.e. a list of columns essentially)
          and hence calculate the performance metric for the a flattened list (i.e. merging all the columns into an
          ordered list such that the order of imputed_values and true_values is matched parfectly).

    :param imputed_values: list of imputed values
    :param true_values: list of true values
    :return:
    """

    # If Categorical Feature
    if isinstance(imputed_values[0], str):

        # Confusion Matrix
        conf_matrix = metrics.confusion_matrix(true_values, imputed_values)

        # Accuracy
        accuracy = metrics.accuracy_score(true_values, imputed_values)

        # Precision
        # precision = metrics.precision_score(true_values, imputed_values)
        # todo: fix - precision not working as it requires a 'positive' label, it could work better if I encoded the categories

        return(conf_matrix, accuracy)

    # If Numerical Feature
    elif isinstance(imputed_values[0], float):

        # Difference Metric D for all values
        D = np.array(true_values) - np.array(imputed_values)

        sample_mean = np.mean(D)
        sample_sd = np.sqrt(np.var(D))

        plt.hist(D)

        return(D, sample_mean, sample_sd)

    else:

        return(None)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### EXTRACT A LIST OF ORIGINAL DATA TO COMPARE WITH IMPUTED DATA

# todo: need to make this into a stand-alone function or incorporate into imputation_methods()
#       the work has started to incorporate it into imputation_methods()

# In order to extract the true dropped data, this loop goes through each column and using a vector
# of boolean inputs slices the dataframe.
columns = ['cat0', 'cat1']
# columns = ['cont0', 'cont1']

true_values = []
for column in columns:
    truefalse_list = (original_df != df)[column].to_numpy().tolist()
    values = original_df[column][truefalse_list]
    print("for column", column, "we have :", len(values), "true values which were dropped")
    true_values.append(values)

# Sample Run of Performance metrics function
results = []

# using true values as imputed as well just as a sanity check
for index, column in enumerate(columns):
    conf_matrix, accuracy = performance_metrics(true_values[index].tolist(), true_values[index].tolist())
    results.append([conf_matrix, accuracy])
    # returns, as expected accuracy 1. and all accurate predicitons.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### SIMULATION STUDY FUNCTION

