---

##### Housekeeping : recall data available and task

We have access to real insurance data. The overall task is to predict a binary class
from several categorical and numeric attributes. In particular:

* *target* is the binary variable to predict
* the *feature columns*, cat0 - cat18 are categorical, and
* the *feature columns* cont0 - cont10 are continuous.

For a total of 18 possible explanatory variables. *Likely won't need to use all.*

**Instructions for our part of the question:**

1. Before making predictions, you should assess three imputation methods by removing (at random)
a small percentage of the data entries, imputing the missing values, and comparing your predictions
to the actual values.

---

## What is Imputation?

> In statistics, imputation is the **process of replacing missing data with substituted values**.

[Wiki](https://en.wikipedia.org/wiki/Imputation_(statistics)

There is quite a lot of terminology and methodologies surrounding what appears to be quite a simple problem.
This is because this issue is not as simple as it may appear. 

##### Terminology:

* When substituting for a data point, it is known as "*unit imputation*";
* When substituting for a component of a data point, it is known as "*item imputation*".

##### Pitfalls of Missing Data:

There are three main problems that missing data causes: 

* Missing data can introduce a substantial amount of **bias**, 

* make the handling and **analysis of the data more arduous**, and

* create **reductions in efficiency**.

##### Common Methods:

There are many theories but the majority introduce bias.
The most well known are:

* **hot deck and cold deck imputation**; 

    *we won't consider this for this task, we have no meaningful ordering of data*.

    One form of hot-deck imputation is called "last observation carried forward"
    (or LOCF for short), which involves sorting a dataset according to any of a
    number of variables, thus creating an ordered dataset. The technique then finds the
    first missing value and uses the cell value immediately prior to the data that are
    missing to impute the missing value.
    
    This method is known to **increase risk of increasing bias** and potentially **false conclusions**.
    Hence it is not recommended.

* **listwise and pairwise deletion**;

    *we won't consider this for this task, we don't care to delete entries*.
    
    It **reduces power** and it consists of simply deleting the rows with missing data.
    
* **mean imputation**;

    *interesting*

    *How*?
    Replacing any missing value with the mean of that variable for all other cases.
    
    *Benefit* : not changing the sample mean for that variable.
    
    *Downside* : mean imputation **attenuates any correlations** involving the variable(s) that are imputed.
    
    *Refining the method*: mean imputation can be carried out within classes (i.e. categories such as gender).
    This is a special case of **generalized regression imputation**.
    
* **non-negative matrix factorization**;

    *interesting*

    Non-negative matrix factorization (NMF) can take missing data while minimizing its cost function,
    rather than treating these missing data as zeros that could introduce biases.
    This makes it a **mathematically proven method for data imputation**.
    
    *How*?
    1. prove that the missing data are ignored in the cost function,
    2. prove that the impact from missing data can be as small as a second order effect.

* **regression imputation**;

    *interesting*

    *How*?
    1. Estimate a regression model and
    2. use fitted values from the model to impute the missing values
    
    *Benefits* : 
    
    *Downside* : there is no uncertainty associated with the missing data - 'too great precision' -
    if you just pick out the corresponding fitted value from the regression line.
    So the regression model predicts the most likely value but does not supply uncertainty about the value.
    This **introduces a lot of bias**.
    
    Any single imputation does not take into account the uncertainty in the imputations.
    After imputation, the data is treated as if they were the actual real values in single imputation.
    The negligence of uncertainty in the imputation can and will lead to overly precise results and errors in any conclusions drawn.
    
    *Refining the method* : stochastic regression manages to correct the lack of an error term in the imputation,
    by adding the average regression variance to the regression imputations to introduce error. Gives **much less bias**,
    but still the noise should be higher.
    
* **multiple imputation**;

    *interesting*
    
    This is a method which aims to average the outcomes across multiple imputed datasets.
    
    *How*?
    1. Imputation – Similar to single imputation, missing values are imputed.
      However, the imputed values are drawn m times from a distribution rather than just once.
      At the end of this step, there should be m completed datasets.

    2. Analysis – Each of the m datasets is analyzed. At the end of this step there should be m analyses.
                                                                                                                                                                                                                                                                                                                                                                                                                
    3. Pooling – The m results are consolidated into one result by calculating the mean, variance,
      and confidence interval of the variable of concern or by combining simulations from each separate model.
    
    *Benefits* : multiple imputation is flexible and can be used in a wide variety of scenarios e.g. when data is *missing
    completely at random*, *missing at random*, and even when the data is *missing not at random*.
    
    *Downside* : multiple imputation accounts for the uncertainty and range of values that the true value could have taken.
    
    The most common variation is Multiple Imputation by Chained Equations (MICE), also known as
    "fully conditional specification" and, "sequential regression multiple imputation."
    
    *Usage*: The MICE package allows users in R to perform multiple imputation using the MICE method.


## Some practical tutorials

[A brief guide to data imputation with Python and R](https://towardsdatascience.com/a-brief-guide-to-data-imputation-with-python-and-r-5dc551a95027)

##### Using R

`MICE` Package, useful bits and bobs:

* `md.pattern()` shows the distribution of missing values and combinations of missing features.

* The Imputer is set up in two steps:
    
    1. **Prepare the Imputer**: choose prediction methods and include/exclude columns from the computation.
    
        *Note* that you can use a different method for each column.
    
        *Methods examples* include, `mean` (unconditional mean), `ppm` (predictive mean matching),
        `norm` (prediction by Bayesian linear regression based on other features, i.e. stochastic regression),
        `logreg` (prediction by logistic regression for 2-value variable). 
    
        *See more in the docs for mice() and methods(your_mice_instance)*
    
    2. **Apply the Imputer**: just run command (*see docs*)

##### Using R

`sklearn` Package, useful bits and bobs:

* The imputer component of the sklearn package has more cool features like
imputation through K-nearest algorithm. (*see docs*)

* `SimpleImputer`, this is a less complicated algorithm with less predictive power.
the use is simple, check out the code on the article.

## From the `sklearn` docs

[Imputing missing values before building an estimator](https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py)

[Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)

Note that the sample code can be found within. A few key points;

1. More on the `IterativeImputer` / `MICE` :

    **IterativeImputer is inspired by R MICE package**

    > In the statistics community, it is common practice to perform multiple imputations, generating, for example,
      m separate imputations for a single feature matrix. Each of these m imputations is then put through the subsequent
      analysis pipeline (e.g. feature engineering, clustering, regression, classification). The m final analysis results
      (e.g. held-out validation errors) allow the data scientist to obtain understanding of how analytic results may differ
      as a consequence of the inherent uncertainty caused by the missing values.
      The above practice is called multiple imputation.

2. More on `Nearest Neighbours Imputation` :

    A **euclidean distance metric** that supports missing values, is used to find the nearest neighbors.
    Each missing feature is imputed using values from n_neighbors nearest neighbors that have a value for the feature.
    The feature of the neighbors are **averaged uniformly or weighted by distance to each neighbor**. 

    > Reference: Troyanskaya et. al, Missing value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17 no. 6, 2001 Pages 520-525.

    


