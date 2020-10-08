"""
resreg: Resampling strategies for regression in Python
"""






#=====================#
# Imports
#=====================#

import numpy as np
import pandas as pd
import copy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import KernelDensity
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor






#=======================================#
# Functions for evaluating relevance
#=======================================#

def sigmoid(y, s, c):
    """
    Sigmoid function for an array (y) with shape and center values, s and c, 
    respectively
    """
    
    y = np.squeeze(np.asarray(y))
    return 1/(1 + np.exp(-s * (y - c)))




def sigmoid_relevance(y, cl, ch):
    """
    Map an array (y) to relevance values (0 to 1) with a sigmoid function.
    
    
    Parameters
    -----------
    y : array_like
        The target values to be mapped to relevance values
    cl : float
        Center of sigmoid for lower extreme. Values less than cl will have relevance 
        values greater than 0.5. If only higher extremes are relevant, set cl=None. 
    ch : float
        Center of sigmoid for higher extreme. Values greater than ch will have relevance 
        values greater than 0.5. If only lower extremes are relevant, set ch=None.
    
    Returns
    ----------
    y_relevance : ndarray
        Relevance values mapped to corresponding to values in y
    """
    
    y = np.squeeze(np.asarray(y))
    offset = 0.001 * np.std(y)    
    if cl is None:
        ch = ch + offset
        sh = np.log(1e4 - 1)/(ch) # shape of sigmoid
        return  sigmoid(y, abs(sh), ch) # High extreme        
    elif ch is None:
        cl = cl - offset
        sl = np.log(1e4 - 1)/(cl) # shape of sigmoid
        return sigmoid(y, -abs(sl), cl) # Low extreme
    else: 
        cl, ch = cl - offset, ch + offset
        sh = np.log(1e4 - 1)/(ch)
        sl = np.log(1e4 - 1)/(cl) 
        return sigmoid(y, -abs(sl), cl) + sigmoid(y, abs(sh), ch)




def pdf_relevance(y, bandwidth=1.0):
    """
    Map an array (y) to relevance values (0 to 1) by taking the inverse of the
    probability density function (PDF). A kernel PDF is fitted to the target values using 
    sklearn.neighbors.KernelDensity, and then the inverse is normalized to a range of 0-1.
    This method may be slow for large datasets.
    
    Parameters
    -----------
    y : array_like
        The target values to be mapped to relevance values
    bandwidth : float
        The bandwith of the kernel. Default is 1.0. Higher values indicate a smoother 
        curve.
    
    Returns
    ----------
    y_relevance : ndarray
        Relevance values mapped to corresponding to values in y
    """
    
    y = np.squeeze(np.asarray(y))
    y = y.reshape(len(y),1)
    pdf = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    pdf.fit(y)
    pdf_vals = np.exp(pdf.score_samples(y))
    y_relevance = 1 - (pdf_vals - pdf_vals.min())/(pdf_vals.max() - pdf_vals.min())
    
    return y_relevance






#===========================================================#
# Functions for validation and evaluation of performance
#===========================================================#

def bin_split(y, bins):
    """ 
    Split target values (y) into bins
    
    Parameters
    ------------
    y : array-like
        The target values
    bins : array_like
        A one-dimensional and monotonically increasing array of boundary values for 
        splitting y into bins.
    
    Returns
    --------
    (bin_indices, bin_freqs)  : tuple
        A tuple containing a list of indices for each bin (bin_indices) and a Pandas 
        dataframe containg the frequency of each bin (bin_freqs).
        
    Example
    --------
    >>> X = np.random.uniform(0, 1, size=(100,10))
    >>> y = np.random.uniform(0, 1, size=100)
    >>> bin_indices, bin_freqs = bin_split(y, bins=[0.2, 0.5, 0.6, 0.8])
    >>> for bin_index in bin_indices:
            X_bin = X[bin_index,:]
            y_bin = y[bin_index]
    
    Splits y into bins the following bins: y ≤ 0.2, 0.2 ≤ y < 0.5, 0.5 ≤ y < 0.6, 
    0.6 ≤ y < 0.8, and 0.8 ≤ y.
    """
    
    y, bins = np.squeeze(np.asarray(y)), np.asarray(bins)
    assert pd.Series(bins).is_monotonic, ('bins must be monotonically increasing such as'
                    ' [1, 3, 5, 7] rather than [1, 5, 7, 3]')
    assert min(y) < min(bins), (f"A value in bins ({min(bins)}) is outside the range of y")
    assert max(y) > max(bins), (f"A value in bins ({max(bins)}) is outside the range of y")
    

    y_digit = np.digitize(y, bins, right=False)
    numbins = max(y_digit) + 1
    bin_indices = [np.where(y_digit==x)[0] for x in range(numbins)]
    
    # Display distribution
    freqs = [len(x) for x in bin_indices]
    percs = np.asarray(freqs)/np.sum(freqs) * 100
    percs = [round(val, 3) for val in percs]
    percs = percs + [100]
    freqs += [np.sum(freqs)]
    bin_ranges = ['y < ' + str(bins[0])] + \
                    ['{0} ≤ y < {1}'.format(bins[i], bins[i+1]) \
                     for i in range(len(bins)-1)] + \
                     ['y ≥ ' + str(bins[-1])] + ['TOTAL']
                    
    bin_freqs = pd.DataFrame([bin_ranges, freqs, percs]).transpose()
    bin_freqs.columns = ['bin range', 'frequency', 'percent']
    
    # Return indices for each bin
    return (bin_indices, bin_freqs)

    


def uniform_test_split(X, y, bins, bin_test_size=0.5, verbose=False, random_state=None):
    """
    Split arrays or matrices into train and test subsets such that the test set is 
    roughly uniform (i.e. having the same number of samples from each bin).
    
    Parameters
    -----------
    X : array-like or matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    bins : array_like
        A one-dimensional and monotonically increasing array of boundary values for 
        splitting y into bins.
    bin_test_size : int or float (default=0.5)
        If int, bin_test_size is the number of samples drawn from each bin to form the 
        uniform test set. If float (between 0 and 1), it is the fraction of the size of 
        the smallest bin.
    verbose : bool, optional (default=False)
        If True, print a dataframe showing the range and the frequency of each bin for
        the training and testing set
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ----------
    [train_indices, test_indices] : list
        List containing train and test indices of data
    
    Example
    ---------
    >>> [train_indices, test_indices] = uniform_test_split(X, y, bins=[20, 40, 55, 87],
    ...                                         bin_test_size=0.4)
    >>> X_train, y_train = X[:,train_indices], y[train_indices]
    >>> X_test, y_test = X[:,test_indices], y[test_indices]
    
        Target values are split into the following bins: y ≤ 20, 20 ≤ y < 40, 40 ≤ y < 55,
        55 ≤ y < 87, and 87 ≤ y. From each bin, 0.4 of the number of samples in the 
        smallest bin are drawn to form the test set, and the remaining samples form the 
        training set.
    """
    
    # Split into bins
    y = np.squeeze(np.asarray(y))
    X = np.asarray(X)
    assert len(X) == len(y), 'X and y must be of the same length.'
    bin_indices, bin_freqs = bin_split(y, bins)
    bin_sizes = [len(index) for index in bin_indices]    
    if 0 < bin_test_size < 1:
        bin_test_size = int(bin_test_size * min(bin_sizes))
    elif bin_test_size in (0.0, 1.0) and type(bin_test_size)==float:
        raise ValueError("bin_test_size (float) must be between 0 and 1")
    
    # Sample training and testing set
    np.random.seed(random_state)
    train_indices, test_indices = [], []
    train_freqs, test_freqs = [], []
    
    for bin_index, bin_size in zip(bin_indices, bin_sizes):
        # Raise errors if needed
        if bin_size <= bin_test_size:
            bin_indices = bin_split(y, bins, verbose=verbose)
            raise ValueError('bin_test_size ({0}) must be smaller than the smallest bin '
                             'which has {1} samples.'.format(bin_test_size, 
                                                    min(bin_sizes)))
            
        test_index = np.random.choice(bin_index, int(bin_test_size), replace=False)
        test_indices.extend(test_index)
        test_freqs.append(len(test_index))
        train_index = set(bin_index) - set(test_index)
        train_indices.extend(list(train_index))
        train_freqs.append(len(train_index))
    train_freqs.append(np.sum(train_freqs))
    test_freqs.append(np.sum(test_freqs))
    
    # Print statistics
    if verbose:
        bin_ranges = ['y < ' + str(bins[0])] + \
                        ['{0} ≤ y < {1}'.format(bins[i], bins[i+1]) \
                         for i in range(len(bins)-1)] + \
                         ['y ≥ ' + str(bins[-1])] + ['TOTAL']
        df = pd.DataFrame([bin_ranges, train_freqs, test_freqs]).transpose()
        df.columns = ['bin range', 'train size', 'test size']
        print('\n')
        print(df.to_string(index=False, justify='right'))
        print('\n')
    
    # Return train and test indices
    return [train_indices, test_indices]




def is_accurate(y_true, y_pred, error_threshold):
    """
    Return 1 if the absolute error is less than or equal to the error threshold, 
    else 0, for all corresponding values in y_true and y_pred
    """
    
    y_true = np.squeeze(np.asarray(y_true))
    y_pred = np.squeeze(np.asarray(y_pred))
    assert len(y_true) == len(y_pred), "y_true and y_pred must be of the same length"
    error = np.abs(y_true - y_pred)
    accurate = np.array(error <= error_threshold, dtype=int)
    return accurate




def accuracy_score(y_true, y_pred, error_threshold, normalize=True):
    """
    Compute the accuracy of predictions in y_pred for corresponding true values in 
    y_true. A prediction is accurate if the absolute error is not more than the 
    error_threshold.
    
    Parameters
    ------------
    y_true : 1d array-like
        True target values
    y_pred : 1d array-like
        Predicted target values, as returned by a regressor
    error_threshold : float
        Maximum absolute error allowed for a prediction to be considered accurate. A 
        prediction is inaccurate if the absolute error is greater than error_threshold.
    normalize : bool, optional (default=True)
        If False, return the number of accuately predicted samples. If True, return the
        fraction of accurately predicted samples.
    
    Returns
    ----------
    score : float
        If normalize is True, score is the number of predictions within an error limit.
        Else, if normalize is False, score is the fraction of predictions within the
        error limit.
    """
    
    accurate = is_accurate(y_true, y_pred, error_threshold)
    score = np.sum(accurate)
    if normalize:
        score = score/len(accurate)
    return score




def accuracy_function(y_true, y_pred, error_threshold, k):
    "Compute the accuracy function as defined in  Torgo and Ribeiro (2009)"
    
    y_true = np.squeeze(np.asarray(y_true))
    y_pred = np.squeeze(np.asarray(y_pred))
    accurate = is_accurate(y_true, y_pred, error_threshold)
    weight = np.abs(y_true - y_pred)
    weight = weight - error_threshold
    weight = np.power(weight, 2)
    weight = -k/error_threshold**2 * weight
    weight = 1 - np.exp(weight)
    return accurate * weight




def precision_score(y_true, y_pred, error_threshold, relevance_pred, relevance_threshold, 
                    k=1e4):
    """
    Compute the precision for regression, as defined by Torgo and Ribeiro (2009)
    
    Parameters
    ------------
    y_true : 1d array-like
        True target values
    y_pred : 1d array-like
        Predicted targed values, as returned by a regressor
    error_threshold : float
        Maximum absolute error allowed for a prediction to be considered accurate. A 
        prediction is inaccurate if the absolute error is greater than error_threshold.
    relevance_pred : 1d array-like
        Relevance of predicted target values
    relevance_threshold : float
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    k : float, optional (default=1e4)
        Value that determines the steepness of the accuracy function.
    
    Returns
    ---------
    precision : float
        A measure of precision for regression problems
    
    References
    -------------
    ..  [1] Torgo, L., and Ribeiro, R. (2009). Precision and recall for regression.
    
    """

    acc_function  = accuracy_function(y_true, y_pred, error_threshold, k)
    relevance = relevance_pred * (relevance_pred >= relevance_threshold)
    precision = np.dot(acc_function, relevance)/np.sum(relevance)
    return precision


    

def recall_score(y_true, y_pred, error_threshold, relevance_true, relevance_threshold,
                 k=1e4):
    """
    Compute the recall for regression, as defined by Torgo and Ribeiro (2009)
    
    Parameters
    ------------
    y_true : 1d array-like
        True target values
    y_pred : 1d array-like
        Predicted targed values, as returned by a regressor
    error_threshold : float
        Maximum absolute error allowed for a prediction to be considered accurate. A 
        prediction is inaccurate if the absolute error is greater than error_threshold.
    relevance_true : 1d array-like
        Relevance of true target values
    relevance_threshold : float
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    k : float, optional (default=1e4)
        Value that determines the steepness of the accuracy function.
    
    Returns
    ---------
    recall : float
        A measure of recall for regression problems
    
    References
    -------------
    ..  [1] Torgo, L., and Ribeiro, R. (2009). Precision and recall for regression.
    """
    
    acc_function  = accuracy_function(y_true, y_pred, error_threshold, k)
    relevance = relevance_true * (relevance_true >= relevance_threshold)
    recall = np.dot(acc_function, relevance)/np.sum(relevance)
    return recall




def f1_score(y_true, y_pred, error_threshold, relevance_true, relevance_pred, 
             relevance_threshold, k=1e4):
    """
    Compute the F1 score for regression, as defined by Torgo and Ribeiro (2009)
    
    Parameters
    ------------
    y_true : 1d array-like
        True target values
    y_pred : 1d array-like
        Predicted targed values, as returned by a regressor
    error_threshold : float
        Maximum absolute error allowed for a prediction to be considered accurate. A 
        prediction is inaccurate if the absolute error is greater than error_threshold.
    relevance_true : 1d array-like
        Relevance of true target values
    relevance_pred: 1d array-like
        Relevance of predicted target values
    relevance_threshold : float
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    k : float (default=1e4)
        Value that determines the steepness of the accuracy function.
    
    Returns
    ---------
    f1score : float
        A measure of recall for regression problems
    
    References
    -------------
    ..  [1] Torgo, L., and Ribeiro, R. (2009). Precision and recall for regression.
    """
    
    precision = precision_score(y_true, y_pred, error_threshold, relevance_pred, 
                                relevance_threshold, k)
    recall = recall_score(y_true, y_pred, error_threshold, relevance_true, 
                          relevance_threshold, k)
    f1score = 2 * precision * recall / (precision + recall)
    if not f1score > 0:
        f1score = 0
    return f1score




def fbeta_score(y_true, y_pred, beta, error_threshold, relevance_true, relevance_pred, 
                relevance_threshold, k=1e4):
    """
    Compute the F-beta score for regression, as defined by Torgo and Ribeiro (2009)
    
    Parameters
    ------------
    y_true : 1d array-like
        True target values
    y_pred : 1d array-like
        Predicted targed values, as returned by a regressor
    beta : float
        Determines the weight of recall in combined score. 
    error_threshold : float
        Maximum absolute error allowed for a prediction to be considered accurate. A 
        prediction is inaccurate if the absolute error is greater than error_threshold.
    relevance_true : 1d array-like
        Relevance of true target values
    relevance_pred: 1d array-like
        Relevance of predicted target values
    relevance_threshold : float
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    k : float, optional (default=1e4)
        Value that determines the steepness of the accuracy function.
    
    Returns
    ---------
    fbeta_score : float
        A measure of recall for regression problems
    
    References
    -------------
    ..  [1] Torgo, L., and Ribeiro, R. (2009). Precision and recall for regression.
    """
    
    precision = precision_score(y_true, y_pred, error_threshold, relevance_pred, 
                                relevance_threshold, k)
    recall = recall_score(y_true, y_pred, error_threshold, relevance_true, 
                          relevance_threshold, k)
    fbeta = (beta**2 + 1) * precision * recall / (beta**2 * precision + recall)
    if not fbeta > 0:
        fbeta = 0
    return fbeta




def matthews_corrcoef(y_true, y_pred, bins):
    """
    Compute the Matthew's correlation coefficient (MCC) by converting the regression
    problem to a multiclass classification problem.
    
    Parameters
    --------------
    y_true : 1d array-like
        True target values
    y_pred : 1d array-like
        Predicted target values, as returned by a regressor
    bins : array_like
        A one-dimensional and monotonically increasing array of boundary values for 
        splitting y into bins.
    
    Returns
    --------
    mcc : float
        The Matthew's correlation coefficient
    
    References
    -----------
    ..  [1] Matthews, B.W. (1975). Comparison of the predicted and observed secondary 
        structure of T4 phage lysozyme.
        [2] Gorodkin, J. (2004). Comparing two K-category assignments by a K-category
        correlation coefficient.
    """
    
    assert len(y_true)==len(y_pred), 'y_true and y_pred must be of the same length.'
    y_true = np.digitize(y_true, bins)
    y_pred = np.digitize(y_pred, bins)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    return mcc




def bin_performance(y_true, y_pred, bins, metric='MSE', error_threshold=None):
    """
    Compute the predictive performance for different bins of the true target values. The
    true target values are split into bins, and the performance on each bin is evaluated.
    
    Parameters
    ------------
    y_true : 1d array-like
        True target values
    y_pred : 1d array-like
        Predicted target values, as returned by a regressor
    bins : array_like
        A one-dimensional and monotonically increasing array of boundary values for 
        splitting y into bins.
    metric : str, {'mse' | 'rmse' | 'mae' | 'accuracy'}, (default='mse')
        Metric to use in computing performance -  'mse': mean squared error, 'rmse': root
        mean squared error, 'mae': mean absolute error, or 'accuracy': accuracy score (as 
        defined in this module, see accuracy_score for details).
    error_threshold : float (default=None)
        Maximum absolute error allowed for a prediction to be considered accurate. 
        A prediction is inaccurate if the absolute error is greater than error_threshold.
        Must be specified if metric='accuracy'. 
        
    Returns
    ---------
    perf_bins : ndarray
        The performance scores (mse, rmse, mae, or accuracy_score) for each bin
    """
    
    y_true, y_pred = np.squeeze(np.asarray(y_true)), np.squeeze(np.asarray(y_pred))
    assert len(y_true)==len(y_pred), 'y_true and y_pred must be of the same length.'
    bin_indices, bin_freqs = bin_split(y_true, bins)
    perf_bins = []
    metric = metric.lower()
    for index in bin_indices:
        y_true_bin, y_pred_bin = y_true[index], y_pred[index]
        if metric=='mse':
            perf = metrics.mean_squared_error(y_true_bin, y_pred_bin)
        elif metric=='rmse':
            perf = metrics.mean_squared_error(y_true_bin, y_pred_bin)
            perf = np.sqrt(perf)
        elif metric=='mae':
            perf = metrics.mean_absolute_error(y_true_bin, y_pred_bin)
        elif metric=='accuracy':
            if error_threshold==None:
                raise ValueError("Must specify error_threshold if metric is 'accuracy'")
            perf = accuracy_score(y_true_bin, y_pred_bin, error_threshold, normalize=True)
        perf_bins.append(perf)
    
    return np.array(perf_bins)






#======================================#
# Functions for resampling datasets
#======================================#

def get_neighbors(X, k):
    """Return indices of k nearest neighbors for each case in X"""
    
    X = np.asarray(X)
    dist = pdist(X)
    dist_mat = squareform(dist)
    order = [np.argsort(row) for row in dist_mat]
    neighbor_indices = np.array([row[1:k+1] for row in order])
    return neighbor_indices




def smoter_interpolate(X, y, k, size, nominal=None, random_state=None):
    """
    Generate new cases by interpolating between cases in the data and a randomly
    selected nearest neighbor. For nominal features, random selection is carried out, 
    rather than interpolation.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    k : int
        Number of nearest neighbors to use in generating synthetic cases by interpolation
    size : int
        Number of new cases to generate
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new cases generated.
        Dimensions of X_new and y_new are the same as X and y, respectively.
   
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    neighbor_indices = get_neighbors(X, k)  # Get indices of k nearest neighbors
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(y)), size, replace=True) 
    X_new, y_new = [], []
        
    for i in sample_indices:
        # Get case and nearest neighbor
        X_case, y_case = X[i,:], y[i]
        neighbor = np.random.choice(neighbor_indices[i,:])
        X_neighbor, y_neighbor = X[neighbor, :], y[neighbor]
        
        # Generate synthetic case by interpolation
        rand = np.random.rand() * np.ones_like(X_case)
        
        if nominal is not None:
            rand = [np.random.choice([0,1]) if x in nominal else rand[x] \
                    for x in range(len(rand))] # Random selection for nominal features, rather than interpolation
            rand = np.asarray(rand)
        diff = (X_case - X_neighbor) * rand
        X_new_case = X_neighbor + diff
        d1 = np.linalg.norm(X_new_case - X_case)
        d2 = np.linalg.norm(X_new_case - X_neighbor)
        y_new_case = (d2 * y_case + d1 * y_neighbor) / (d2 + d1 + 1e-10) # Add 1e-10  to avoid division by zero
        X_new.append(X_new_case)
        y_new.append(y_new_case)
    
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    
    return [X_new, y_new]




def add_gaussian(X, y, delta, size, nominal=None, random_state=None):
    """
    Generate new cases  by adding Gaussian noise to the dataset (X, y) . For nominal 
    features, selection is carried out with weights equal to the probability of the 
    nominal feature.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    delta : float
        Value that determines the magnitude of Gaussian noise added
    size : int
        Number of new cases to generate
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new cases generated.
        Dimensions of X_new and y_new are the same as X and y, respectively.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(y)), size, replace=True)
    stds_X, std_y = np.std(X, axis=0), np.std(y)
    X_sel, y_sel = X[sample_indices,:], y[sample_indices]
    noise_X = np.array([[np.random.normal(loc=0.0, scale=std*delta) for std in stds_X] \
                         for row in range(X_sel.shape[0])])
    noise_y = np.random.normal(loc=0.0, scale=std_y*delta, size=y_sel.shape)
    X_new = X_sel + noise_X
    y_new  = y_sel + noise_y
    
    # Deal with nominal features (selection with weights, not addition of noise)
    if nominal is not None:
        for i in range(X_sel.shape[1]):
            if i in nominal:
                nom_vals, nom_freqs = np.unique(X[:, i], return_counts=True)
                nom_freqs = nom_freqs/nom_freqs.sum()
                nom_select = np.random.choice(nom_vals, size=X_sel.shape[0], p=nom_freqs,
                                              replace=True)
                X_new[:,i] = nom_select
            
    return [X_new, y_new]




def wercs_oversample(X, y, relevance, size, random_state=None):  
    """
    Generate new cases by selecting samples from the original dataset using the 
    relevance as weights. Samples with with high relevance are more likely to be selected
    for oversampling.
    
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : array_like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    size : int
        Number of new cases to generate
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random
    
    
   Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new cases generated.
        Dimensions of X_new and y_new are the same as X and y, respectively.
    """
    
    X, y,  = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    assert len(X)==len(y), 'X and y must be of the same length.'
    assert len(y)==len(relevance), 'y and relevance must be of the same length'
    prob = np.abs(relevance/np.sum(relevance))  # abs to remove very small negative values
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(X)), size=size, p=prob, replace=True)
    X_new, y_new = X[sample_indices,:], y[sample_indices]
    
    return X_new, y_new




def wercs_undersample(X, y, relevance, size, random_state=None):
    """Undersample dataset by removing samples selected using the relevance as weights.
    Samples with low relevance are more likely to be removed in undersampling.
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : array_like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    size : int
        Number of samples in new undersampled dataset (i.e. after removing samples)
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of cases after 
        removing samples. Dimensions of X_new and y_new are the same as X and y, 
        respectively.
    """
    
    X, y,  = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    assert len(X)==len(y), 'X and y must be of the same length.'
    assert 0 < size < len(y), 'size must be smaller than the length of y'
    assert len(y)==len(relevance), 'y and relevance must be of the same length'
    prob = 1 - relevance
    prob = abs(prob/prob.sum())   # abs to remove very small negative numbers
    remove = len(y) - size
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(X)), size=remove, p=prob, replace=False)
    sample_indices = list(set(range(len(X))) - set(sample_indices))
    X_new, y_new = X[sample_indices,:], y[sample_indices]
    
    return X_new, y_new
    



def undersample(X, y, size, random_state=None):
    """
    Randomly undersample a dataset (X, y), and return a smaller dataset (X_new, y_new). 
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    size : int
        Number of samples in new undersampled dataset.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random
    
    Returns
    ----------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) after undersampling.
    """
        
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    if size >= len(y):
        raise ValueError('size must be smaller than the length of y')
    np.random.seed(random_state)
    new_indices = np.random.choice(range(len(y)), size, replace=False)
    X_new, y_new = X[new_indices, :], y[new_indices]
    return [X_new, y_new]  

  

        
def oversample(X, y, size, method, k=None, delta=None, relevance=None, nominal=None,
               random_state=None):
    """
    Randomly oversample a dataset (X, y) and return a larger dataset (X_new, y_new)
    according to specified method.
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    size : int
        Number of samples in new oversampled dataset.
    method : str, {'random_oversample' | 'smoter' | 'gaussian' | 'wercs' | 'wercs-gn'}
        Method for generating new samples. 
        
        If 'random_oversample', samples are duplicated.
        
        If 'smoter', new synthetic samples are generated by interpolation with the SMOTER
        algorithm. 
        
        If 'gaussian', new synthetic samples are generated by addition of Gaussian noise. 
        
        If 'wercs', relevance values are used as weights to select values for duplication. 
        
        If 'wercs-gn', values are selected with relevance values as weights and then 
        Gaussian noise is added.
        
    k : int (default=None)
        Number of nearest neighbors to use in generating synthetic cases by interpolation.
        Must be specified if method is 'smoter'. 
    delta : float (default=None)
        Value that determines the magnitude of Gaussian noise added. Must be specified if
        method is 'gaussian'
    relevance : array_like (default=None)
        Values ranging from 0 to 1 that indicate the relevance of target values. Must be
        specified if method is 'wercs' or 'wercs-gn'
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random
    
    Returns
    ----------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new samples after
        oversampling.
    """
    
    # Prepare data 
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    moresize = int(size - len(y))
    if moresize <=0:
        raise ValueError('size must be larger than the length of y')
    
    
    # Generate extra samples for oversampling
    np.random.seed(seed=random_state)
    if method=='duplicate':
        more_indices = np.random.choice(np.arange(len(y)), moresize, replace=True)
        X_more, y_more = X[more_indices,:], y[more_indices]
        
    elif method=='smoter':
        if k is None:
            raise ValueError("Must specify k if method is 'smoter'")
        [X_more, y_more] = smoter_interpolate(X, y, k, size=moresize, nominal=nominal, 
                                              random_state=random_state)
    
    elif method=='gaussian':
        if delta is None:
            raise ValueError("Must specify delta if method is 'gaussian'")
        [X_more, y_more] = add_gaussian(X, y, delta, size=moresize, nominal=nominal,
                                        random_state=random_state)
    
    elif method=='wercs' or method=='wercs-gn':
        if relevance is None:
            raise ValueError("Must specify relevance if method is 'wercs' or 'wercs-gn'")
        else:
            assert len(y)==len(relevance), 'y and relevance must be of the same length'
            
        [X_more, y_more] = wercs_oversample(X, y, relevance, size=moresize, 
                                            random_state=random_state)
        if method=='wercs-gn':
            if delta is None:
                raise ValueError("Must specify delta if method is 'wercs-gn'")
           
            [X_more, y_more] = add_gaussian(X_more, y_more, delta, size=moresize, 
                                            nominal=nominal, random_state=random_state)
    else:
        raise ValueError('Wrong method specified.')
    
    # Combine old dataset with extrasamples
    X_new = np.append(X, X_more, axis=0)
    y_new = np.append(y, y_more, axis=0)
    
    return [X_new, y_new]


 
        
def split_domains(X, y, relevance, relevance_threshold):
    """
    Split a dataset (X,y) into rare and normal domains according to the relevance of the
    target values. Target values with relevance values below the relevance threshold
    form the normal domain, while other target values form the rare domain.
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : array_like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    
    Returns
    -----------
    [X_norm, y_norm, X_rare, y_rare] : list
        List containing features (X_norm, X_rare) and target values (y_norm, y_rare) of 
        the normal and rare domains.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    assert len(X) == len(y) == len(relevance), ('X, y, and relevance must have the same '
                                                  'length')
    rare_indices = np.where(relevance >= relevance_threshold)[0]
    norm_indices = np.where(relevance < relevance_threshold)[0]
    assert len(rare_indices) < len(norm_indices), ('Rare domain must be smaller than '
              'normal domain. Adjust your relevance values or relevance threshold so '
              'that the there are fewer samples in the rare domain.')
    X_rare, y_rare = X[rare_indices,:], y[rare_indices]
    X_norm, y_norm = X[norm_indices,:], y[norm_indices]
    
    return [X_norm, y_norm, X_rare, y_rare]
    

    



#===========================================#
# Functions to implement resampling methods
#===========================================#
 
def random_undersample(X, y, relevance, relevance_threshold=0.5, under='balance',
                       random_state=None):
    """
    Resample imbalanced dataset by undersampling. The dataset is split into a rare and 
    normal domain using relevance values. Target values with relevance below the relevance
    threshold form the normal domain, and other target values form the rare domain. The 
    normal domain is randomly undersampled and the rare domain is left intact.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    under : float or str, {'balance' | 'extreme' | 'average'} (default='balance')
        Value that determines the amount of undersampling. If float, under is the fraction
        of normal samples removed in undersampling. Half of normal samples are removed if
        under=0.5. 
        
        Otherwise, if 'balance', the normal domain is undersampled so that it has the same
        number of samples as the rare domain. 
        
        If 'extreme', the normal domain is undersampled so that the ratio between the 
        sizes of the normal and rare domain is inverted. 
        
        If 'average', the extent of undersampling is intermediate between 'balance' and 
        'extreme'.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
        
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)

    # Determine size of normal domain after undersampling
    if type(under)==float:
        assert 0 < under < 1, "under must be between 0 and 1"
        new_norm_size = int((1 - under) * len(y_norm))
    elif under=='balance':
       new_norm_size = int(len(y_rare))
    elif under=='extreme':
         new_norm_size = int(len(y_rare)**2 / len(y_norm))
         if new_norm_size <= 1:
             raise ValueError("under='extreme' results in a normal domain with {0} "
                              "samples".format(new_norm_size))
    elif under=='average':
        new_norm_size = int((len(y_rare) + len(y_rare)**2 / len(y_norm)) / 2)
    else:
        raise ValueError("Incorrect value of 'under' specified.")
   
    # Undersample normal domain
    [X_norm_new, y_norm_new] = undersample(X_norm, y_norm, size=new_norm_size, 
                                            random_state=random_state)
    X_new = np.append(X_norm_new, X_rare, axis=0)
    y_new = np.append(y_norm_new, y_rare, axis=0)
    
    return [X_new, y_new]
        
    
    
    
def random_oversample(X, y, relevance, relevance_threshold=0.5, over='balance', 
                      random_state=None):
    """
    Resample imbalanced dataset by oversampling. The dataset is split into a rare and 
    normal domain using relevance values. Target values with relevance below the relevance
    threshold form the normal domain, and other target values form the rare domain. The 
    rare domain is randomly oversampled (duplicated) and the normal domain is left intact.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    over : float or str, {'balance' | 'extreme' | 'average'} (default='balance')
        Value that determines the amount of oversampling. If float, over is the fraction
        of rare samples duplicated in oversampling. Half of rare samples are duplicated if
        over=0.5. 
        
        Otherwise, if 'balance', the rare domain is oversampled so that it has the same 
        number of samples as the normal domain. 
        
        If 'extreme', the rare domain is oversampled so that the ratio between the sizes 
        of the normal and rare domain is inverted. 
        
        If 'medium', the extent of oversampling is intermediate between 'balance' and 
        'extreme'.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)
    
    # Determine size of rare domain after oversampling
    if type(over)==float:
        assert over >=0 , "over must be non-negative"
        new_rare_size = int((1 + over) * len(y_rare))
    elif over=='balance':
       new_rare_size = len(y_norm)
    elif over=='extreme':
         new_rare_size = int(len(y_norm)**2 / len(y_rare))
    elif over=='average':
        new_rare_size = int((len(y_norm) + len(y_norm)**2 / len(y_rare)) / 2)
    else:
        raise ValueError('Incorrect value of over specified')
   
    # Oversample rare domain
    [X_rare_new, y_rare_new] = oversample(X_rare, y_rare, size=new_rare_size,
                                            method='duplicate', random_state=random_state)
    X_new = np.append(X_norm, X_rare_new, axis=0)
    y_new = np.append(y_norm, y_rare_new, axis=0)
    
    return [X_new, y_new]




def smoter(X, y, relevance, relevance_threshold=0.5, k=5, over='balance', under=None, 
		   nominal=None, random_state=None):
    """
    Resample imbalanced dataset with the SMOTER algorithm. The dataset is split into a 
    rare normal domain using relevance values. Target values with relevance below the 
    relevance threshold form the normal domain, and other target values form the rare 
    domain. The rare domain is oversampled by interpolating between samples, and the 
    normal domain is undersampled.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    k : int (default=5)
        Number of nearest neighbors to use in generating synthetic cases by interpolation.
    over : float or str, {'balance' | 'extreme' | 'average'} (default='balance')
        Value that determines the amount of oversampling. If float, over is the fraction
        of new rare samples generated in oversampling.  
        
        Otherwise, if string, over indicates the amount of both oversampling and 
        undersampling. If 'balance', the rare domain is oversampled and the normal domain 
        is undersampled so that they are equal in size.
        
        If 'extreme', oversampling and undersampling are done so that the ratio of the 
        sizes of rare domain to normal domain is inverted. 
        
        If 'average' the extent of oversampling and undersampling is intermediate between 
        'balance' and 'extreme'.
    under : float (default=None)
        Value that determines the amount of undersampling. Should only be specified if
        over is float. One-third of normal samples are removed if under=0.33.
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Torgo, L. et al (2015). Resampling strategies for regression. 
        [2] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression.       
    """
    
    # Split data into rare and normal dormains
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)
    norm_size, rare_size = len(y_norm), len(y_rare)
    
    # Determine new sizes for rare and normal domains after oversampling
    if type(over)==float:
        assert type(under)==float, 'under must also be a float if over is a float'
        assert 0 <= under <= 1, 'under must be between 0 and 1'
        assert over >=0 , "over must be non-negative"
        new_rare_size = int((1 + over) * rare_size)
        new_norm_size = int((1 - under) * norm_size)
    elif over=='balance':
        new_rare_size = new_norm_size = int((norm_size + rare_size)/2)
    elif over == 'extreme':
        new_rare_size, new_norm_size = norm_size, rare_size
    elif over == 'average':
        new_rare_size = int(((norm_size + rare_size)/2 + norm_size)/2)
        new_norm_size = int(((norm_size + rare_size)/2 + rare_size)/2)
    else:
        raise ValueError("Incorrect value of over, must be a float or  "
                         "'balance', 'extreme', or 'average'")
        
    # Oversample rare domain
    y_median = np.median(y)
    low_indices = np.where(y_rare < y_median)[0]
    high_indices = np.where(y_rare >= y_median)[0]
    
    # First oversample low rare cases
    if len(low_indices) != 0:
        size = int(len(low_indices)/rare_size * new_rare_size)
        X_low_rare, y_low_rare = oversample(X_rare[low_indices,:], y_rare[low_indices], 
                                     size=size, method='smoter', k=k, relevance=relevance,
                                     nominal=nominal, random_state=random_state)
        
    # Then do high rare cases
    if len(high_indices) != 0:
        size = int(len(high_indices)/rare_size * new_rare_size)
        X_high_rare, y_high_rare = oversample(X_rare[high_indices], y_rare[high_indices],
                                     size=size, method='smoter', k=k, relevance=relevance,
                                     nominal=nominal, random_state=random_state)
    
    # Combine oversampled low and high rare cases
    if min(len(low_indices), len(high_indices)) != 0:
        X_rare_new = np.append(X_low_rare, X_high_rare, axis=0)
        y_rare_new = np.append(y_low_rare, y_high_rare, axis=0)
    elif len(low_indices) == 0:
        X_rare_new =  X_high_rare
        y_rare_new =  y_high_rare
    elif len(high_indices) == 0:
        X_rare_new =  X_low_rare
        y_rare_new = y_low_rare
        
    # Undersample normal cases
    X_norm_new, y_norm_new = undersample(X_norm, y_norm, size=new_norm_size, 
                                         random_state=random_state)
    
    # Combine resampled rare and normal cases
    X_new = np.append(X_rare_new, X_norm_new, axis=0)
    y_new = np.append(y_rare_new, y_norm_new, axis=0)
    
    return (X_new, y_new)




def gaussian_noise(X, y, relevance, relevance_threshold=0.5, delta=0.05, over=None, under=None,
                   nominal=None, random_state=None):
    """
    Resample imbalanced dataset by introduction of Gaussian noise. The dataset is split 
    into a rare and normal domain using relevance values. Target values with relevance 
    below the relevance threshold form the normal domain, and other target values form the
    rare domain. The rare domain is oversampled by addition of Gaussian noise, and the 
    normal domain is undersampled.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    delta : float (default=0.05)
        Value that determines the magnitude of Gaussian noise added
    over : float or str, {'balance' | 'medium' | 'extreme'} (default='balance')
        Value that determines the amount of oversampling. If float, over is the fraction
        of new rare samples generated in oversampling.  
        
        Otherwise, if string, over indicates the amount of both oversampling and 
        undersampling. If 'balance', the rare domain is oversampled and the normal domain 
        is undersampled so that they are equal in size.
        
        If 'extreme', oversampling and undersampling are done so that the ratio of the 
        sizes of rare domain to normal domain is inverted. 
        
        If 'medium' the extent of oversampling and undersampling is intermediate between 
        'balance' and 'extreme'.
    under : float (default=None)
        Value that determines the amount of undersampling. Should only be specified if
        over is float. One-third of normal samples are removed if under=0.33.
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression. 
    """
    
    # Split data into rare and normal dormains
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)
    norm_size, rare_size = len(y_norm), len(y_rare)
    
    # Determine new sizes for rare and normal domains after oversampling
    if type(over)==float:
        assert type(under)==float, 'under must be a float if over is a float'
        assert 0 <= under <= 1, 'under must be between 0 and 1'
        assert over >=0 , "over must be non-negative"
        new_rare_size = int((1 + over) * rare_size)
        new_norm_size = int((1 - under) * norm_size)
    elif over=='balance':
        new_rare_size = new_norm_size = int((norm_size + rare_size)/2)
    elif over == 'extreme':
        new_rare_size, new_norm_size = norm_size, rare_size
    elif over == 'average':
        new_rare_size = int(((norm_size + rare_size)/2 + norm_size)/2)
        new_norm_size = int(((norm_size + rare_size)/2 + rare_size)/2)
    else:
        raise ValueError("Incorrect value of over specified, must be a float or  "
                         "'balance', 'extreme', or 'average'")
        
    # Oversample rare domain
    y_median = np.median(y)
    low_indices = np.where(y_rare < y_median)[0]
    high_indices = np.where(y_rare >= y_median)[0]
    
    # First oversample low rare cases
    if len(low_indices) != 0:
        size = int(len(low_indices)/rare_size * new_rare_size)
        X_low_rare, y_low_rare = oversample(X_rare[low_indices,:], y_rare[low_indices], 
                                           size=size, method='gaussian', delta=delta, 
                                           relevance=relevance, nominal=nominal, 
                                           random_state=random_state)
        
    # Then do high rare cases
    if len(high_indices) != 0:
        size = int(len(high_indices)/rare_size * new_rare_size)
        X_high_rare, y_high_rare = oversample(X_rare[high_indices], y_rare[high_indices],
                                     size=size, method='gaussian', delta=delta, 
                                     relevance=relevance, nominal=nominal, 
                                     random_state=random_state)
    
    # Combine oversampled low and high rare cases
    if min(len(low_indices), len(high_indices)) != 0:
        X_rare_new = np.append(X_low_rare, X_high_rare, axis=0)
        y_rare_new = np.append(y_low_rare, y_high_rare, axis=0)
    elif len(low_indices) == 0:
        X_rare_new =  X_high_rare
        y_rare_new =  y_high_rare
    elif len(high_indices) == 0:
        X_rare_new =  X_low_rare
        y_rare_new = y_low_rare
        
    # Undersample normal cases
    X_norm_new, y_norm_new = undersample(X_norm, y_norm, size=new_norm_size, 
                                         random_state=random_state)
    
    # Combine resampled rare and normal cases
    X_new = np.append(X_rare_new, X_norm_new, axis=0)
    y_new = np.append(y_rare_new, y_norm_new, axis=0)
    
    return (X_new, y_new)




def wercs(X, y, relevance, over=0.5, under=0.5, noise=False, delta=0.05, nominal=None,
          random_state=None):
    """
    Resample imbalanced dataset with the WERCS algorithm. The relevance values are used
    as weights to select samples for oversampling and undersampling such that samples with
    high relevance are more likely to be selected for oversampling and less likely to be
    selected for undersampling.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    over : float (default=0.5)
        Fraction of new samples generated in oversampling.
    under : float (default=0.5)
        Fraction of samples removed in undersampling.
    noise : bool
        Whether to add Gaussian noise to samples selected for oversampling (wercs-gn).
    delta : float (default=0.05)
        Value that determines the magnitude of Gaussian noise added.
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    over_size = int(over * len(y))
    under_size = int((1 - under) * len(y))
    X_over, y_over = wercs_oversample(X, y, relevance=relevance, size=over_size, 
                                      random_state=random_state) # Oversample
    X_under, y_under = wercs_undersample(X, y, relevance=relevance, size=under_size,
                                         random_state=random_state)
    if noise:
        X_under, y_under = add_gaussian(X_under, y_under, delta=delta, size=under_size,
                                        nominal=nominal, random_state=random_state)
    X_new = np.append(X_over, X_under, axis=0)
    y_new = np.append(y_over, y_under, axis=0)
    
    return [X_new, y_new]
    





#================================================================#
# Combining ensemble learning (bagging) and resampling methods
#================================================================#

class Rebagg():
    """
    Rebagg implements resampled bagging to deal with data imbalance in regression 
    problems. Each regressor in the bagging ensemble is fitted to a resampled dataset. 
    Resampling may be implemented by random oversampling, SMOTER, addition of Gaussian 
    noise, or the WERCS algorithm.
    
    Parameters
    ------------
    m : int (default=100)
        Number of models (regressors) in the ensemble
    s : int or float (default=0.5)
        Number of bootstrap samples drawn to fit the base regressors. If int, then *s* 
        samples are drawn. If float, then *X.shape[0] * s* samples are drawn.
    base_reg : scikit-learn regressor or None, optional (default=None)
        The base regressor to fit random subsets of the dataset. If None, then *base_reg* 
        is a decision tree with default settings.
   
    Attributes
    ------------
    fitted_regs : list
        A list of base regressors which have been fitted to bootstrap samples of the 
        dataset.
    pred_std : ndarray
        An array of the standard deviation of predicted values after calling the predict
        method.
        
    References
    -----------
    ..  [1] Branco, P., Torgo, L., Ribeiro, R.P. (2018). Rebagg: Resampled bagging for 
        imbalanced regression.
    
    Examples
    ----------
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> knreg = KNeighborsRegressor(n_neighbors=10)
    >>> #
    >>> # Initialize an ensemble of 50 KNN regressors, with each base regressor fitted to
    >>> # a resampled dataset of 200 samples
    >>> #
    >>> rebagg = Rebagg(m=50, s=200, base_reg=knreg) 
    >>> rebagg.fit(X_train, y_train) # Resample dataset and fit base regressors
    >>> y_pred = rebagg.predict(X_test)
    >>> y_std = rebagg.pred_std  # Standard deviation of ensemble predictions
    
    """
    
    
    
    
    def __init__(self, m=100, s=0.5, base_reg=None):
        self.m = m
        if type(s) not in [int, float]:
            raise TypeError("s must be int or float")
        self.s = s
        if base_reg is not None:
            self.base_reg = base_reg
        else:
            self.base_reg = DecisionTreeRegressor()
        self._isfitted = False
        
        

    
    def sample(self, X, y, relevance, relevance_threshold, 
               sample_method='random_oversample', size_method='balance', k=5, delta=0.1, 
               over=0.5, under=0.5, nominal=None, random_state=None):
        """ 
        Resample dataset and return a smaller balanced dataset for fitting a single 
        regressor in the ensemble.
        
        Parameters
        ------------
        X : array-like or sparse matrix 
            Features of the data.
        y : array-like
            The target values.
        relevance : 1d array-like
            Values ranging from 0 to 1 that indicate the relevance of target values. 
        relevance_threshold : float
            Threshold of relevance for forming rare and normal domains. Target values with 
            relevance less than relevance_threshold form the normal domain, while target 
            values with relevance greater than or equal to relevance_threshold form the 
            rare domain.
        sample_method : str, {'random_oversample'|'smoter'|'gaussian'|'wercs'|'wercs-gn'},
                        (default='random_oversample')
            Method for generating new samples. 
        
            If 'random_oversample', a random selection of rare samples are duplicated.
        
            If 'smoter', new rare samples are generated by interpolation with the SMOTER 
            algorithm. 
        
            If 'gaussian', new rare samples are generated by addition of Gaussian noise. 
        
            If 'wercs', relevance values are used as weights to select samples for both 
            oversampling and undersampling. 
        
            If 'wercs-gn', relevance values are used as weights to select samples for both 
            oversampling and undersampling, and then gaussian noise is added.
            
        size_method : str, {'balance' | 'variation'} (default=balance)
            Method for selecting samples from dataset for fitting each regressor in 
            ensemble. Must be specified unless sample_method is 'wercs' or 'wercs-gn'. 
            
            If 'balance', equal number of samples are chosen from the rare and normal 
            domains (i.e. s/2 from each). 
            
            If 'variation', the s * p and s * (1 - p) samples are chosen from the rare and
            normal domains, respectively. p is randomly chosen from 
            [1/3, 2/5, 1/2, 2/5, 2/3].
            
        k : int (default=5)
            Number of nearest neighbors to use in generating synthetic cases by 
            interpolation. Necessary if sample_method is 'smoter'.
        delta : float (default=0.1)
            Value that determines the magnitude of Gaussian noise added. Necessary if
            method is 'gaussian' or 'wercs-gn'
        over : float (default=0.5)
            Value that determines the amount of oversampling if sample_method is 'wercs'
            or 'wercs-gn'. Indicates the fraction of new samples generated in 
            oversampling.
        under : float (default=0.5)
            Value that determines the amount of undersampling if sample_method is 'wercs'
            or 'wercs-gn'.  Indicates the fraction of samples removed in undersampling.
        nominal : ndarray (default=None)
            Column indices of nominal features. If None, then all features are continuous.
        random_state : int or None, optional (default=None)
            If int, random_state is the seed used by the random number generator. If None, 
            the random number generator is the RandomState instance used by np.random.
        
    
        
        Returns
        --------
        [X_reg, y_reg] : list
            List contanining features (X_reg) and target values (y_reg) of resampled 
            dataset for fitting a single regressor in the ensemble.
        """
        
        X, y = np.asarray(X), np.squeeze(np.asarray(y))
        relevance = np.squeeze(np.asarray(relevance))
        assert len(y)==len(X), ('X and y do not have the same number of samples')
        assert len(y)==len(relevance), 'y and relevance must be of the same length'
        
        if type(self.s) == float:
            size = int(self.s * X.shape[0])
        elif type(self.s) == int:
            size = self.s
            
        np.random.seed(seed=random_state)
        
        if sample_method in ('random_oversample', 'smoter', 'gaussian'):
            # Split target values to rare and normal domains
            X_norm_all, y_norm_all, X_rare_all, y_rare_all = split_domains(X, y, 
                            relevance=relevance, relevance_threshold=relevance_threshold)
            relevance_rare = relevance[np.where(relevance >= relevance_threshold)]
           
            # Sample sizes
            if size_method=='balance':
                s_rare = int(size/2)
                s_norm = size - s_rare
            elif size_method=='variation':
                p = np.random.choice([1/3, 2/5, 1/2, 3/5, 2/3])
                s_rare = int(size * p)
                s_norm = int(size - s_rare)
            else:
                raise ValueError('Wrong value of size_method specified')
            
            # Sample rare data
            if s_rare <= len(y_rare_all):
                rare_indices = np.random.choice(range(len(y_rare_all)), s_rare, 
                                               replace=False) # No oversampling
                X_rare, y_rare = X_rare_all[rare_indices,:], y_rare_all[rare_indices]
            else:
                if sample_method=='random_oversample':
                    rare_indices = np.random.choice(range(len(y_rare_all)), s_rare,
                                                    replace=True) # Duplicate samples
                    X_rare, y_rare = X_rare_all[rare_indices,:], y_rare_all[rare_indices]
                elif sample_method=='smoter':
                    X_rare, y_rare = oversample(X_rare_all, y_rare_all, size=s_rare,
                                                method='smoter', k=k, 
                                                relevance=relevance_rare, nominal=nominal,
                                                random_state=random_state)
                elif sample_method=='gaussian':
                    X_rare, y_rare = oversample(X_rare_all, y_rare_all, size=s_rare,
                                                method='gaussian', delta=delta, 
                                                relevance=relevance_rare, nominal=nominal,
                                                random_state=random_state)
            # Sample normal data
            norm_indices = np.random.choice(range(len(y_norm_all)), s_norm, replace=True)
            X_norm, y_norm = X[norm_indices,:], y[norm_indices]
            
            # Combine rare and normal samles
            X_reg = np.append(X_rare, X_norm, axis=0)
            y_reg = np.append(y_rare, y_norm, axis=0)
            
        elif sample_method in ['wercs', 'wercs-gn']:
            noise = True if sample_method=='wercs-gn' else False
            X_reg, y_reg = wercs(X, y, relevance=relevance, over=over, under=under, 
                                 noise=noise, delta=delta, nominal=nominal, 
                                 random_state=random_state)
            sample_indices = np.random.choice(range(len(y_reg)), size=size, 
                                              replace=(size>len(y_reg)))
            X_reg, y_reg = X_reg[sample_indices,:], y_reg[sample_indices]
        
        else:
            raise ValueError("Wrong value of sample_method")
        
        
        return [X_reg, y_reg]
    
    
    
    
    def fit(self, X, y, relevance, relevance_threshold=0.5, sample_method='random_oversample',
            size_method='balance', k=5, delta=0.1, over=0.5, under=0.5, nominal=None, 
            random_state=None):
        """ 
        Fit an ensemble of regressors to randomly drawn samples from the training set.
        
        Parameters
        ------------
        X : array-like or sparse matrix 
            Features of the data.
        y : array-like
            The target values.
        relevance : 1d array-like
            Values ranging from 0 to 1 that indicate the relevance of target values. 
        relevance_threshold : float (default=0.5)
            Threshold of relevance for forming rare and normal domains. Target values with 
            relevance less than relevance_threshold form the normal domain, while target 
            values with relevance greater than or equal to relevance_threshold form the rare
            domain. Must be specified unless sample_method is 'wercs' or 'wercs-gn'.
        sample_method : str, {'random_oversample'|'smoter'|'gaussian'|'wercs'|'wercs-gn'},
                        (default='random_oversample')
            Method for generating new samples. 
        
            If 'random_oversample', a random selection of rare samples are duplicated.
        
            If 'smoter', new rare samples are generated by interpolation with the SMOTER 
            algorithm. 
        
            If 'gaussian', new rare samples are generated by addition of Gaussian noise. 
        
            If 'wercs', relevance values are used as weights to select samples for both 
            oversampling and undersampling. 
        
            If 'wercs-gn', relevance values are used as weights to select samples for both 
            oversampling and undersampling, and then gaussian noise is added.
            
        size_method : str, {'balance' | 'variation'} (default=None)
            Method for selecting samples from dataset for fitting each regressor in 
            ensemble. Must be specified unless sample_method is 'wercs' or 'wercs-gn'. 
            
            If 'balance', equal number of samples are chosen from the rare and normal 
            domains (i.e. s/2 from each). 
            
            If 'variation', the s * p and s * (1 - p) samples are chosen from the rare and
            normal domains, respectively. p is randomly chosen from 
            [1/3, 2/5, 1/2, 2/5, 2/3].
            
        k : int (default=None)
            Number of nearest neighbors to use in generating synthetic cases by 
            interpolation. Must be specified if sample_method is 'smoter'.
        delta : float (default=None)
            Value that determines the magnitude of Gaussian noise added. Must be specified
            if method is 'gaussian'
        over : float (default=None)
            Value that determines the amount of oversampling if sample_method is 'wercs'
            or 'wercs-gn'. Indicates the fraction of new samples generated in 
            oversampling.
        under : float (default=None)
            Value that determines the amount of undersampling if sample_method is 'wercs'
            or 'wercs-gn'.  Indicates the fraction of samples removed in undersampling.
        nominal : ndarray (default=None)
            Column indices of nominal features. If None, then all features are continuous.
        random_state : int or None, optional (default=None)
            If int, random_state is the seed used by the random number generator. If None, 
            the random number generator is the RandomState instance used by np.random.
        
        Returns
        --------
        None
        
        Examples
        ---------
        >>> # Derive relevance values from the probability density function
        >>> relevance = pdf_relevance(y_train)  
        >>> # Ensemble of 50 decision trees, each fitted to 100 samples
        >>> rebagg = Rebagg(m=50, s=500)
        >>> # Fit ensemble to training data, and resample with SMOTER method
        >>> rebagg.fit(X_train, y_train, relevance=relevance, relevance_threshold=0.5,
                       sample_method='smoter', size_method='balance', k=5)
        """
        
        regressors = [copy.deepcopy(self.base_reg) for i in range(self.m)]
        np.random.seed(seed=random_state)
        random_states = np.random.choice(range(2*self.m), size=self.m)
        for i in range(self.m):
            # Bootstrap samples
            X_reg, y_reg = self.sample(X, y, relevance, relevance_threshold, sample_method,
                                       size_method, k, delta, over, under, nominal,
                                       random_states[i])
            # Fit regressor to samples
            regressors[i].fit(X_reg, y_reg)
        self.fitted_regs = regressors
        self._isfitted=True
        
        
        
        
    def predict(self, X):
        """
        Predict target values for X. The predicted target value is the mean of all values
        predicted by each regressor in the ensemble. The standard deviation of the 
        predicted values can be obtained by the attribute, `pred_std`.
        
        Parameters
        -----------
        X : array-like or sparse matrix 
            Features of the data.
        
        Returns
        --------
        y : array-like of shape (n_samples,)
            The predicted target values
        
        Examples
        ----------
        >>> y_pred = rebagg.predict(X_test)
        >>> pred_std = rebagg.pred_std  # Standard deviation of predictions
        """
        if not self._isfitted:
            raise ValueError('Rebagg ensemble has not yet been fitted to training data.')
        y_ensemble = np.array([reg.predict(X) for reg in self.fitted_regs])
        y = np.mean(y_ensemble, axis=0)
        self.pred_std = np.std(y_ensemble, axis=0)
        assert len(y)==len(X), ('X and y do not have the same number of samples')
        
        return y



        
#========================================================================================#    
