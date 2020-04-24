**Resampling for regression (resreg)**
========================================

Resreg is a Python package for resampling imbalanced distributions in regression problems.

If you find resreg useful, please cite all of the following:

1. Gado, J.E., Beckham, G.T., and Payne, C.M (2020). Improving enzyme optimum temperature prediction with resampling strategies and ensemble learning.
2. Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for imbalanced distributions in regression.
3. Branco, P., Torgo, L., and Ribeiro, R.P. (2018). REBAGG: Resampled bagging for imbalanced regression.


Installation
-------------
Install with pip

.. code:: shell-session

    pip install resreg

Or from source

.. code:: shell-session

    git clone https://github.com/jafetgado/resreg.git
    cd resreg
    python setup.py install



Prerequisites
-------------

1. Python 3
2. Numpy
3. Scipy
4. Pandas
5. Scikit-learn


Usage
-----
A regression dataset (X, y) can be resampled to mitigate the imbalance in the distribution with any of six strategies: random oversampling, random undersampling, SMOTER, Gaussian noise, WERCS, or Rebagg.

1. Random oversampling: randomly oversamples rare values selected by the user via a relevance function.
2. Random undersampling: randomly undersamples abundant values.
3. SMOTER: randomly undersamples abundant values; oversamples rare values by interpolation between near neighbors.
4. Gaussian noise: randomly undersamples abundant values; oversamples rare values by adding Gaussian noise.
5. WERCS: resamples the dataset by selecting instances using user-specified relevance values as weights.
6. Rebagg: Trains an ensemble of base learners on independently resampled bootstrap subsets of the dataset.

See the tutorial for more details.


Examples
----------
.. code:: python

    import resreg
    from sklearn.metrics import train_test_split
    from sklearn.metrics import RandomForestRegressor

    # Split dataset to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Resample training set with random oversampling
    relevance = resreg.sigmoid_relevance(y, cl=None, ch=np.percentile(y, 90))
    X_train, y_train = resreg.random_oversampling(X_train, y_train, relevance, relevance_threshold=0.5,
                                                  over='balance')

    # Fit regressor to resampled training set
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train, y_train)
