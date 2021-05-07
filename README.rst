**Resampling for regression (resreg)**
========================================

Resreg is a Python package for resampling imbalanced distributions in regression problems.

If you find resreg useful, please cite the following article:

* Gado, J.E., Beckham, G.T., and Payne, C.M (2020). Improving enzyme optimum temperature prediction with resampling strategies and ensemble learning. *J. Chem. Inf. Model.* 60(8), 4098-4107.

If you use RO, RU, SMOTER, GN, or WERCS methods, also cite

* Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for imbalanced distributions in regression. *Neurocomputing.* 343, 76-99.

If you use REBAGG, also cite

* Branco, P., Torgo, L., and Ribeiro, R.P. (2018). REBAGG: Resampled bagging for imbalanced regression. In *2nd International Workshop on Learning with Imbalanced Domains: Theory and Applications.* pp 67-81.

If you use precision, recall, or F1-score for regression, also cite

* Torgo, L. and Ribeira, R.P. (2009). Precision and recall for regression. In *International Conference on Discovery Science.* pp332-346


Installation
-------------
Preferrably, install from GitHub source. The use of a virtual environment is strongly advised.

.. code:: shell-session

    git clone https://github.com/jafetgado/resreg.git
    cd resreg
    pip install -r requirements.txt
    python setup.py install


Or, install with pip (less preferred)

.. code:: shell-session

    pip install resreg



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

1. Random oversampling (RO): randomly oversample rare values selected by the user via a relevance function.
2. Random undersampling (RU): randomly undersample abundant values.
3. SMOTER: randomly undersample abundant values; oversample rare values by interpolation between nearest neighbors.
4. Gaussian noise (GN): randomly undersample abundant values; oversample rare values by adding Gaussian noise.
5. WERCS: resample the dataset by selecting instances using user-specified relevance values as weights.
6. REBAGG: Train an ensemble of Scikit-learn base learners on independently resampled bootstrap subsets of the dataset.

See the tutorial for more details.


Examples
----------
.. code:: python

    import resreg
    from sklearn.metrics import train_test_split
    from sklearn.metrics import RandomForestRegressor

    # Split dataset to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Resample training set with random oversampling such that values above the
    # the 90th percentile are equal size with other values (balance)
    relevance = resreg.sigmoid_relevance(y, cl=None, ch=np.percentile(y, 90))
    X_train_res, y_train_res = resreg.random_oversampling(X_train, y_train, relevance,
                                                          relevance_threshold=0.5,
                                                          over='balance')

    # Fit regressor to resampled training set
    reg = RandomForestRegressor()
    reg.fit(X_train_res, y_train_res)
    y_pred = reg.predict(X_test, y_test)
