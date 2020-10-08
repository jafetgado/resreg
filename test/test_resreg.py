"""
 Test resreg           
"""

import pytest
import numpy as np
import pandas as pd
import copy
from sklearn.neighbors import KNeighborsRegressor

import sys
sys.path.insert(1, '../../resreg/')
import resreg









# Fixtures
#=====================#

def different_array_types(array):
    return [np.asarray(array), list(array), tuple(array),
            pd.Series(array), pd.DataFrame(array)]

            
np.random.seed(0)
y_mixed = np.random.normal(0, 1, 100)  # both negative and positive values
pos_and_neg_params = different_array_types(y_mixed)
@pytest.fixture(params=pos_and_neg_params)
def positive_and_negative_array(request):
    return request.param


np.random.seed(0)
y_neg = np.random.normal(-5, 1, 100)
neg_params = different_array_types(y_neg)
@pytest.fixture(params=neg_params)
def negative_array(request):
    return request.param


np.random.seed(0)
y_pos = np.random.normal(5, 1, 100)
pos_params = different_array_types(y_pos)
@pytest.fixture(params=pos_params)
def positive_array(request):
    return request.param






# Test relevance functions
#===============================#
def sum_sigmoid_relevance(y, low_percentile, high_percentile):
    cl = np.percentile(y, low_percentile) if low_percentile is not None else None
    ch = np.percentile(y, high_percentile) if high_percentile is not None else None
    relevance = resreg.sigmoid_relevance(y, cl, ch)
    return round(np.sum(relevance), 3)


def sum_pdf_relevance(y, bandwidth=1.0):
    relevance = resreg.pdf_relevance(y, bandwidth)
    return round(np.sum(relevance), 3)




def test_sigmoid():    
    assert set([resreg.sigmoid(10, x, 10) for x in [0.01, 0.1, 1, 10, 100]]) == {0.5}
    



@pytest.mark.parametrize('positive_and_negative_array, negative_array, positive_array',
                    zip(pos_and_neg_params, neg_params, pos_params), 
                    indirect=True)
def test_sigmoid_relevance(positive_and_negative_array, negative_array, positive_array):
    assert sum_sigmoid_relevance(positive_and_negative_array, 10, 90) == 21.393
    assert sum_sigmoid_relevance(positive_and_negative_array, 10, None) == 11.174
    assert sum_sigmoid_relevance(positive_and_negative_array, None, 90) == 10.219
    assert sum_sigmoid_relevance(negative_array, 10, 90) == 32.921
    assert sum_sigmoid_relevance(negative_array, 10, None) == 20.073
    assert sum_sigmoid_relevance(negative_array, None, 90) == 12.848
    assert sum_sigmoid_relevance(positive_array, 10, 90) == 33.634
    assert sum_sigmoid_relevance(positive_array, 10, None) == 15.015
    assert sum_sigmoid_relevance(positive_array, None, 90) == 18.619

  

   
@pytest.mark.parametrize('positive_and_negative_array, negative_array, positive_array',
                    zip(pos_and_neg_params, neg_params, pos_params), 
                    indirect=True)
def test_pdf_relevance(positive_and_negative_array, negative_array, positive_array):
    assert sum_pdf_relevance(positive_and_negative_array) == 22.776
    assert sum_pdf_relevance(negative_array) == 22.776
    assert sum_pdf_relevance(positive_array) == 22.776






# Test validation functions
#===============================#

np.random.seed(0)
y_true = np.random.normal(0, 1, 100)  
y_pred = y_true - np.random.normal(0, 0.5, 100)
X = np.random.normal(0, 1, (100,5))
bins_correct = [np.percentile(y_true, x) for x in [10, 33, 67, 90]]
bins_wrong_1 = [min(y_true) - 1] + bins_correct # outside y_true range
bins_wrong_2 = bins_correct + [max(y_true) + 1] # outside y_true range
bins_wrong_3 = [np.percentile(y_true, x) for x in [10, 33, 90, 67]] # not monotonic




def indices_sum(bin_indices):
    return [np.sum(bin_index) for bin_index in bin_indices]




@pytest.mark.parametrize("bins_correct, bins_wrong_1, bins_wrong_2, bins_wrong_3",
                         [(bins_correct, bins_wrong_1, bins_wrong_2, bins_wrong_3)])
def test_bin_split(bins_correct, bins_wrong_1, bins_wrong_2, bins_wrong_3):
    bin_indices, bin_freqs = resreg.bin_split(y_true, bins_correct)
    assert indices_sum(bin_indices) == [467, 1273, 1699, 1127, 384]
    with pytest.raises(Exception):
        assert resreg.bin_split(y_true, bins_wrong_1)
    with pytest.raises(Exception):
        assert resreg.bin_split(y_true, bins_wrong_2)
    with pytest.raises(Exception):
        assert resreg.bin_split(y_true, bins_wrong_3)




def test_bin_split_array_type(positive_and_negative_array):
    '''Test that bin_split works fine on np.array, pd.Series, pd.DataFrame, lists, 
    and tuples.'''
    bin_indices, bin_freqs = resreg.bin_split(positive_and_negative_array, bins_correct)
    assert indices_sum(bin_indices) == [467, 1273, 1699, 1127, 384]

    

    
@pytest.mark.parametrize("size, sum_tuple",
                         [(0.5, (3615, 1335)), (1, (4674, 276)), (4, (3795, 1155))])
def test_uniform_test_split(size, sum_tuple):
    [train_indices, test_indices] = resreg.uniform_test_split(X, y_true, 
                                            bins=bins_correct, bin_test_size=size, 
                                            verbose=False, random_state=0)
    assert (np.sum(train_indices), np.sum(test_indices)) == sum_tuple
    
    with pytest.raises(Exception):
        [train_indices, test_indices] = resreg.uniform_test_split(X, y_true, 
                                            bins=bins_correct, bin_test_size=1.0, 
                                            verbose=False, random_state=0)
    with pytest.raises(Exception):
        [train_indices, test_indices] = resreg.uniform_test_split(X, y_true, 
                                            bins=bins_correct, bin_test_size=11, 
                                            verbose=False, random_state=0)



def test_is_accurate():
    assert np.sum(resreg.is_accurate(y_true, y_pred, 0.25)) == 28
    with pytest.raises(Exception):
        resreg.is_accurate(y_true, y_pred[:-1], 0.25)




def test_accuracy_score():
    assert resreg.accuracy_score(y_true, y_pred, 0.25, normalize=True) == 0.28
    assert resreg.accuracy_score(y_true, y_pred, 0.25, normalize=False) == 28
    
    


def test_accuracy_function():
    function_sum = np.sum(resreg.accuracy_function(y_true, y_pred, 0.25, 10))
    assert round(function_sum, 3) == 18.463




def test_precision_recall_f1():
    error_threshold = 0.25
    relevance_threshold = 0.5
    k = 10
    beta = 0.5
    relevance_true = resreg.pdf_relevance(y_true)
    relevance_pred = resreg.pdf_relevance(y_pred)
    precision = resreg.precision_score(y_true, y_pred, error_threshold, relevance_pred, 
                                       relevance_threshold, k)
    recall = resreg.recall_score(y_true, y_pred, error_threshold, relevance_true, 
                                       relevance_threshold, k)
    f1 = resreg.f1_score(y_true, y_pred, error_threshold, relevance_true, relevance_pred, 
                         relevance_threshold, k)
    fbeta = resreg.fbeta_score(y_true, y_pred, beta, error_threshold, relevance_true, relevance_pred, 
                         relevance_threshold, k)
    assert round(precision, 3) == 0.227
    assert round(recall, 3) == 0.274
    assert round(f1, 3) == 0.248
    assert round(fbeta, 3) == 0.235




def test_matthews_corrcoef():
    mcc = resreg.matthews_corrcoef(y_true, y_pred, bins_correct)
    assert round(mcc, 3) == 0.453




def test_bin_performance():
    error_threshold = 0.25
    bin_error_mse = resreg.bin_performance(y_true, y_pred, bins_correct, 'MsE')
    bin_error_rmse = resreg.bin_performance(y_true, y_pred, bins_correct, 'RmSE')
    bin_error_mae = resreg.bin_performance(y_true, y_pred, bins_correct, 'MaE')
    bin_error_acc = resreg.bin_performance(y_true, y_pred, bins_correct, 'accURacy',
                                           error_threshold)
    assert round(np.sum(bin_error_mse), 3) == 1.320
    assert round(np.sum(bin_error_rmse), 3) == 2.563
    assert round(np.sum(bin_error_mae), 3) == 2.170
    assert round(np.sum(bin_error_acc), 3) == 1.402
    with pytest.raises(Exception):
        bin_error_acc = resreg.bin_performance(y_true, y_pred, bins_correct, 'accuracy') 






# Tests for resampling functions
#====================================#

np.random.seed(0)
y = np.random.normal(0, 1, 100)  
X = np.random.normal(0, 1, (100,5))
relevance = resreg.pdf_relevance(y)




def test_get_neighbors():
    neighbor_indices = resreg.get_neighbors(X, k=5)
    assert np.sum(neighbor_indices[0,:]) == 191




def test_smoter_interpolate():
    X_new1, y_new1 = resreg.smoter_interpolate(X, y, k=5, size=5, nominal=None, 
                                               random_state=0)    
    X_, y_ = copy.deepcopy(X), copy.deepcopy(y)
    np.random.seed(0)
    X_[:,2:] = np.random.choice([0,1,2], size=(X_.shape[0],3))
    X_new2, y_new2 = resreg.smoter_interpolate(X_, y_, k=5, size=5, nominal=[2,3,4],
                                               random_state=0)
    assert round(np.sum(y_new1), 3) == 1.255
    assert round(np.sum(y_new2), 3) == 0.582




def test_add_gaussian():
    X_new1, y_new1 = resreg.add_gaussian(X, y, delta=0.1, size=5, nominal=None, 
                                         random_state=0) 
    X_, y_ = copy.deepcopy(X), copy.deepcopy(y)
    np.random.seed(0)
    X_[:,2:] = np.random.choice([0,1,2], size=(X_.shape[0],3))
    X_new2, y_new2 = resreg.add_gaussian(X_, y_, delta=0.1, size=5, nominal=[2,3,4],
                                         random_state=0)
    assert round(np.sum(y_new1), 3) == 1.222
    assert round(np.sum(y_new2), 3) == 1.222
    assert round(np.sum(np.sum(X_new1, axis=0)), 3) == 0.841
    assert round(np.sum(np.sum(X_new2, axis=0)), 3) == 14.858




def test_wercs_oversample():
    X_new, y_new = resreg.wercs_oversample(X, y, relevance, size=5, random_state=0)
    assert round(np.sum(y_new), 3) == -2.986
    
    with pytest.raises(Exception):
        X_new, y_new = resreg.wercs_oversample(X, y, relevance[:-1], size=5)
    with pytest.raises(Exception):
        X_new, y_new = resreg.wercs_oversample(X, y[:-1], relevance, size=5, random_state=0)    




def test_wercs_undersample():
    X_new, y_new = resreg.wercs_undersample(X, y, relevance, size=10, random_state=0)
    assert round(np.sum(y_new), 3) == 2.465
    
    with pytest.raises(Exception):
        X_new, y_new = resreg.wercs_undersample(X, y, relevance[:-1], size=10)
    with pytest.raises(Exception):
        X_new, y_new = resreg.wercs_undersample(X, y[:-1], relevance, size=10)
    with pytest.raises(Exception):
        X_new, y_new = resreg.wercs_undersample(X, y, relevance, size=120)
    



def test_undersample():
    X_new, y_new = resreg.undersample(X, y, size=10, random_state=0)
    assert round(np.sum(y_new), 3) == 3.861
    
    with pytest.raises(Exception):
       X_new, y_new = resreg.undersample(X, y, size=100, random_state=0) 




def test_oversample():
    # duplicate
    X_new, y_new = resreg.oversample(X, y, size=120, method='duplicate', random_state=0)
    assert round(np.sum(y_new), 3) == 3.437
    
    # smoter
    X_new, y_new = resreg.oversample(X, y, size=120, method='smoter', k=5, random_state=0)
    assert round(np.sum(y_new), 3) == 4.532
    with pytest.raises(Exception):
        X_new, y_new = resreg.oversample(X, y, size=120, method='smoter', random_state=0)
    
    # gaussian
    X_new, y_new = resreg.oversample(X, y, size=120, method='gaussian', delta=0.1, 
                                     random_state=0)
    assert round(np.sum(y_new), 3) == 2.745
    with pytest.raises(Exception):
        X_new, y_new = resreg.oversample(X, y, size=120, method='gaussian')
        
    # wercs and wercs-gn
    X_new, y_new = resreg.oversample(X, y, size=120, method='wercs', relevance=relevance,
                                     random_state=0)
    assert round(np.sum(y_new), 3) == 12.395
    X_new, y_new = resreg.oversample(X, y, size=120, method='wercs-gn', delta=0.1,
                                     relevance=relevance, random_state=0)
    assert round(np.sum(y_new), 3) == 14.455
    with pytest.raises(Exception):
        X_new, y_new = resreg.oversample(X, y, size=120, method='wercs')
    with pytest.raises(Exception):
        X_new, y_new = resreg.oversample(X, y, size=120, method='wercs-gn', 
                                         relevance=relevance)

    # wrong method
    with pytest.raises(Exception):
        X_new, y_new = resreg.oversample(X, y, size=120, method='nonsense', 
                                         relevance=relevance)
        



def test_split_domains():
    X_norm, y_norm, X_rare, y_rare = resreg.split_domains(X, y, relevance=relevance,
                                                          relevance_threshold=0.5)
    assert round(min(y_norm), 3) == -1.253
    assert round(max(y_norm), 3) == 1.494
    assert round(np.sum(y_rare), 3) == -0.313
    assert round(np.sum(y_norm), 3) == 6.294
    





# Tests for resampling strategies
#====================================#

np.random.seed(0)
y = np.random.normal(0, 1, 100)  
X = np.random.normal(0, 1, (100,5))
relevance = resreg.pdf_relevance(y)
bins = [np.percentile(y_true, x) for x in [10, 33, 67, 90]]




def test_random_undersample():
    # float 
    X_new, y_new = resreg.random_undersample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, under=0.75, 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == 2.041
    
    # balance
    X_new, y_new = resreg.random_undersample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, under='balance', 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == 0.847
    
    # extreme
    X_new, y_new = resreg.random_undersample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, under='extreme', 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == -0.266
    
    # average
    X_new, y_new = resreg.random_undersample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, under='average', 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == -0.746
    
    # errors
    with pytest.raises(Exception):
        X_new, y_new = resreg.random_undersample(X, y, relevance=relevance, 
                                                 relevance_threshold=0.5, under=2)
    with pytest.raises(Exception):
        X_new, y_new = resreg.random_undersample(X, y, relevance=relevance, 
                                                 relevance_threshold=0.8, under='extreme')
    with pytest.raises(Exception):
        X_new, y_new = resreg.random_undersample(X, y, relevance=relevance, 
                                                 relevance_threshold=0.8, under='nonsense')
        
        




def test_random_oversample():
    # float 
    X_new, y_new = resreg.random_oversample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, over=0.75, 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == 4.971
    
    # balance
    X_new, y_new = resreg.random_oversample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, over='balance', 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == 5.090
    
    # extreme
    X_new, y_new = resreg.random_oversample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, over='extreme', 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == -49.722
    
    # average
    X_new, y_new = resreg.random_oversample(X, y, relevance=relevance, 
                                             relevance_threshold=0.5, over='average', 
                                             random_state=0)
    assert round(np.sum(y_new), 3) == -41.166
    
    # errors
    with pytest.raises(Exception):
        X_new, y_new = resreg.random_oversample(X, y, relevance=relevance, 
                                                 relevance_threshold=0.5, over=2)




def test_smoter():
    # float
    X_new, y_new = resreg.smoter(X, y, relevance, relevance_threshold=0.5, k=5, over=0.5,
                                 under=0.5, random_state=0)
    assert round(np.sum(y_new), 3) == 0.612
    
    # balance
    X_new, y_new = resreg.smoter(X, y, relevance, relevance_threshold=0.5, k=5, 
                                     over='balance', random_state=0)
    assert round(np.sum(y_new), 3) == 3.488
    
    # extreme
    X_new, y_new = resreg.smoter(X, y, relevance, relevance_threshold=0.5, k=5, 
                                     over='extreme', random_state=0)
    assert round(np.sum(y_new), 3) == 1.862
    
    # average
    X_new, y_new = resreg.smoter(X, y, relevance, relevance_threshold=0.5, k=5, 
                                     over='average', random_state=0)
    assert round(np.sum(y_new), 3) == 3.658
    # errors
    with pytest.raises(Exception):
        X_new, y_new = resreg.smoter(X, y, relevance, relevance_threshold=0.5, k=5, 
                                     over=0.5, under=1)
    with pytest.raises(Exception):
        X_new, y_new = resreg.smoter(X, y, relevance, relevance_threshold=0.5, k=5, 
                                     over=0.5, under=1.5)
    with pytest.raises(Exception):
        X_new, y_new = resreg.smoter(X, y, relevance, relevance_threshold=0.5, k=5, 
                                     over='balancee', under=1.5)
       



def test_gaussian_noise():
    # float
    X_new, y_new = resreg.gaussian_noise(X, y, relevance, relevance_threshold=0.5, 
                                         delta=0.1, over=0.5, under=0.5, random_state=0)
    assert round(np.sum(y_new), 3) == 0.450
    
    # balance
    X_new, y_new = resreg.gaussian_noise(X, y, relevance, relevance_threshold=0.5, 
                                        delta=0.1, over='balance', random_state=0)
    assert round(np.sum(y_new), 3) == 2.504
    
    # extreme
    X_new, y_new = resreg.gaussian_noise(X, y, relevance, relevance_threshold=0.5, 
                                        delta=0.1, over='extreme', random_state=0)
    assert round(np.sum(y_new), 3) == 4.541
    
    # average
    X_new, y_new = resreg.gaussian_noise(X, y, relevance, relevance_threshold=0.5, 
                                        delta=0.1, over='average', random_state=0)
    assert round(np.sum(y_new), 3) == 5.669
    
    # errors
    with pytest.raises(Exception):
        X_new, y_new = resreg.gaussian_noise(X, y, relevance, relevance_threshold=0.5, 
                                        delta=0.1, over=0.5, under=1)
    with pytest.raises(Exception):
        X_new, y_new = resreg.gaussian_noise(X, y, relevance, relevance_threshold=0.5, 
                                        delta=0.1, over=0.5, under=1.5)
    with pytest.raises(Exception):
        X_new, y_new = resreg.gaussian_noise(X, y, relevance, relevance_threshold=0.5, 
                                        delta=0.1, over=0, under=1)
       


def test_wercs():
    # without noise (wercs)
    X_new, y_new = resreg.wercs(X, y, relevance, over=0.5, under=0.5, random_state=0)
    assert round(np.sum(y_new), 3) == 15.584
    
    # add noise (wercs-gn)
    X_new, y_new = resreg.wercs(X, y, relevance, over=0.5, under=0.5, noise=True, 
                                delta=0.1, random_state=0)
    assert round(np.sum(y_new), 3) == 15.307
    



class RebaggTests():
    global rebagg1
    global rebagg2
    global rebagg3
    rebagg1 = resreg.Rebagg()
    knr = KNeighborsRegressor()
    rebagg2 = resreg.Rebagg(m=20, s=0.33, base_reg=knr)
    rebagg3 = resreg.Rebagg(m=20, s=10.50, base_reg=None)
    
    
    
    def test_init(self):
        with pytest.raises(Exception):
            rebagg_wrong = resreg.Rebagg(s='balance')
    
    
    def test_sample(self):
        X_ens1, y_ens1 = rebagg1.sample(X, y, relevance, relevance_threshold=0.5, 
                                sample_method='random_oversample', size_method='balance', 
                                random_state=0)   
        X_ens2, y_ens2 = rebagg2.sample(X, y, relevance, relevance_threshold=0.5, 
                                sample_method='smoter', k=5, size_method='balance', 
                                random_state=0) 
        X_ens3, y_ens3 = rebagg3.sample(X, y, relevance, relevance_threshold=0.5, 
                                sample_method='gaussian', delta=0.1, 
                                size_method='variation', random_state=0)   
        X_ens4, y_ens4 = rebagg3.sample(X, y, relevance, relevance_threshold=0.5, 
                                sample_method='wercs', over=0.5, under=0.5, random_state=0)
        X_ens5, y_ens5 = rebagg3.sample(X, y, relevance, relevance_threshold=0.5, 
                                sample_method='wercs-gn', delta=0.1, over=0.5, under=0.5,
                                random_state=0)
        
        assert len(y_ens1) == 50
        assert len(y_ens2) == 33
        assert len(y_ens3) == len(y_ens4) == len(y_ens5) == 1050
        assert round(np.sum(y_ens1), 3) == -7.992
        assert round(np.sum(y_ens2), 3) == -1.326
        assert round(np.sum(y_ens3), 3) == -45.111
        assert round(np.sum(y_ens4), 3) == 171.763
        assert round(np.sum(y_ens5), 3) == 157.665
        
        with pytest.raises(Exception):
            X_ens, y_ens = rebagg1.sample(X, y, relevance, relevance_threshold=0.5, 
                                          sample_method='wrong')
        
        
    def test_fit_and_predict(self):
        rebagg1.fit(X, y, relevance, relevance_threshold=0.5, 
                    sample_method='random_oversample', size_method='balance', 
                    random_state=0)
        y_pred1 = rebagg1.predict(X)
        
        
        rebagg2.fit(X, y, relevance, relevance_threshold=0.5, sample_method='smoter', k=1,
                    size_method='balance', random_state=0) 
        y_pred2 = rebagg2.predict(X)
        
        
        rebagg3.fit(X, y, relevance, relevance_threshold=0.5, sample_method='gaussian', 
                    delta=0.1, size_method='variation', random_state=0) 
        y_pred3 = rebagg3.predict(X)
        
        
        rebagg3.fit(X, y, relevance, relevance_threshold=0.5, sample_method='wercs', 
                    over=0.5, under=0.5, random_state=0)
        y_pred4 = rebagg3.predict(X)
        
        
        rebagg3.fit(X, y, relevance, relevance_threshold=0.5, 
                    sample_method='wercs-gn', delta=0.1, over=0.5, under=0.5, 
                    random_state=0)
        y_pred5 = rebagg3.predict(X)
        
        assert round(np.sum(y_pred1), 3) == -7.060
        assert round(np.sum(y_pred2), 3) == -25.532
        assert round(np.sum(y_pred3), 3) == 3.388
        assert round(np.sum(y_pred4), 3) == 8.213
        assert round(np.sum(y_pred5), 3) == 12.123
        
#========================================================================================#