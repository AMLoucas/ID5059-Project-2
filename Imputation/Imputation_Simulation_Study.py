# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### IMPORT LIBRARIES

# essentials
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys as sys

# for modelling
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

# for performance
from sklearn import metrics
from scipy.stats import norm
import random
import time

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

    :param df: pandas DataFrame which is the full data
    :param columns: list of columns to 'drop' values from. Note: must be string name of columns
    :param percentage: float input such as 0.20 (i.e. 20%)
    :return: DataFrame with NaN entries
    """

    start = time.time()
    # Set up: Focus on subset of data dependent on columns,
    #         hence extract information about subset and number of values to drop
    subset = df[columns].copy()
    nrow, ncol = df[columns].shape
    n_samples = int(percentage*nrow)

    # 1. Pick out a vector of random samples (row indexes)
    row_indexes = np.arange(0, nrow, 1).tolist()

    # 2. Sample with replacement from the list of possible indexes
    #    note, `choices` is used to sample with replacement
    sampled_row_indexes = random.choices(row_indexes, k = n_samples)
    sampled_columns = random.choices(columns, k = n_samples)

    # true_value = []
    for i in range(0, n_samples):
        # true_value.append([subset.loc[sampled_row_indexes[i], sampled_columns[i]], sampled_row_indexes[i], sampled_columns[i]])
        if sampled_columns[i][0:3] == 'cat':
            subset.loc[sampled_row_indexes[i], sampled_columns[i]] = 'nan'
        else:
            subset.loc[sampled_row_indexes[i], sampled_columns[i]] = -1
            # this is the convention for a number of sklearn packages

    # note having it as a loop ensures we are not holding a massive dataframe in memory so its a lot faster.
    # e.g. the vectorised version subset.loc[sampled_row_indexes, sampled_columns] = np.nan is too expensive

    # 4. Update data frame with new NaN values
    df.update(subset)
    df = df.replace('nan', np.nan)
    df = df.replace(-1, np.nan)

    end = time.time()
    print("Random Sampling Complete - Time Elapsed : ", end - start)

    return(df)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### FUNCTION TO RUN MODELS

# todo: still does not support discrete

def univariate_imputation_methods(df):
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
    median_imputation_data = df.copy()
    median_imputer = SimpleImputer(strategy = "median")
    median_imputer.fit(median_imputation_data)

    # Once the data has been impute is outputs in an array
    RawOutput = median_imputer.transform(median_imputation_data)

    # We want to convert the array back to data frame now for further computations.
    median_imputation_data = pd.DataFrame(RawOutput,
                                     columns = median_imputation_data.columns,
                                     index = median_imputation_data.index)

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

    """
    ### STRATEGY 3: KNNImputer

    knn_imputer_data = df.copy()

    # We can adjust hyperparameter n_neighbors.
    knn_imputer = KNNImputer(n_neighbors = 2, weights = "uniform")

    # Imputing values
    RawOutput = knn_imputer.fit_transform(knn_imputer_data)

    # We want to convert the array back to data frame now for further computations.
    knn_imputer_data = pd.DataFrame(RawOutput,
                                     columns = knn_imputer_data.columns,
                                     index = knn_imputer_data.index)

    end = time.time()

    print("KNN Imputation Complete - Time Elapsed :", end - middle2)
    """

    return(median_imputation_data, miss_forest_data)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### PERFORMANCE METRICS FUNCTION

def performance_metrics(imputed_values, true_values):
    """
    Function to calculate all performance metrics of interest for each list of imputed vs true values.

    VERY IMPORTANT: **each list for now is specific to each column**.
    NOTE: depending the on the methods used the columns are independent of each other, e.g. the median imputation does not
          use any other column at all - implying that we can assess performance independently of one another.

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

        # 1. Raw bias (RB)
        #    The difference between the expected value of the estimate and truth
        RB = np.array(imputed_values) - np.array(true_values)
        sample_mean = np.mean(RB)
        sample_sd = np.sqrt(np.var(RB))

        #    Sexy Plot
        ci = norm(*norm.fit(RB)).interval(0.95)  # fit a normal distribution and get 95% c.i.
        height, bins, patches = plt.hist(RB, alpha = 0.3)
        plt.fill_betweenx([0, height.max()], ci[0], ci[1], color = 'g', edgecolor = 'white', alpha = 0.1)  # Mark between 0 and the highest bar in the histogram

        # 2. Percentage bias (PB)
        PB = 100 * abs((np.array(imputed_values) - np.array(true_values))/np.array(true_values))

        # 3. Coverage rate (CR).
        #    The coverage rate (CR) is the proportion of confidence intervals that contain the true value.

        # 4. Average width (AW)
        #    The average width of the confidence interval is an indicator of statistical efficiency.

        # 5. RMSE
        #    While the RMSE is widely used, it is not a suitable metric to evaluate multiple imputation methods.

        return(RB, PB, sample_mean, sample_sd)

    else:

        return(None)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### SIMULATION STUDY FUNCTION

def imputation_complete(complete_df, columns, percentage):
    """
    Function which brings together all the functions for a single step of the simulation study.
    These consists of:
    1. randomly dropping values from the dataset,
    2. fitting imputation methods, and
    3. evaluating the performance of the imputed data against the true data.

    :param df: pandas DataFrame which is the full data
    :param columns: list of columns to 'drop' values from. Note: must be string name of columns
    :param percentage: float input such as 0.20 (i.e. 20%)
    :return: results for each of the methods (currently only missforest and median imputer)
    """

    ##### COMPLETE CODE:

    ### 1. DROP VALUES RANDOMLY

    df_nans = remove_random_values(complete_df.copy(), columns, percentage)

    ### 2. COMPUTE METHODS

    median_imputed_data, missforest_imputed_data = univariate_imputation_methods(df_nans[columns])

    ### 3. EXTRACT TRUE VALUES AND COMPARE WITH IMPUTED VALUES

    results_median = []
    results_missforest = []
    for column in columns:

        # a. Extract boolean vector (True/False) to identify Imputed True Data Vectors
        boolean_values = (original_df != df_nans)[column].to_numpy().tolist()
        values = original_df[column][boolean_values]

        # b. Store away true value and boolean vector
        true_values = values

        # c. Extract Imputed Data Vectors
        imputed_values_median = median_imputed_data[column][boolean_values].to_numpy().tolist()
        imputed_values_missforest = missforest_imputed_data[column][boolean_values].to_numpy().tolist()

        # d. Evaluate performance metrics
        results_median.append(performance_metrics(imputed_values_median, true_values))
        results_missforest.append(performance_metrics(imputed_values_missforest, true_values))

    return(results_median, results_missforest)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### DEMO : RUN THE WHOLE THING

# Import Data
original_df = pd.read_csv("data/train.csv")
original_df.head(6)

### 1. CONTINUOUS CASE WITH 20% of TOTAL DATA

start = time.time()

results_median, results_missforest = imputation_complete(complete_df = original_df, columns = ['cont5', 'cont8'], percentage = 0.2)

end = time.time()
print("Full Run Completed - Time Elapsed :", end - start)

# Time Elapsed : 3 minutes

### 2. DISCRETE CASE ONLY WITH 20% of TOTAL DATA




