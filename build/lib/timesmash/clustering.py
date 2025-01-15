import pandas as pd
import numpy as np
from timesmash import Quantizer, XHMMFeatures, InferredHMMLikelihood
from sklearn.cluster import KMeans
from multiprocessing import Pool
from timesmash.utils import callwithValidKwargs, RANDOM_NAME
from timesmash.cynet import fit_helper, _check_data, debug_mpi
from itertools import repeat, product, chain
import pickle

def is_equal_labels(l1,l2):
    ss1 = []
    for lb, dataframe in l1.groupby(l1.columns[0]):
        ss1.append(frozenset(dataframe.index))
    ss2 = []
    for lb, dataframe in l2.groupby(l2.columns[0]):
        ss2.append(frozenset(dataframe.index))
    return frozenset(ss1) == frozenset(ss2)

class XHMMClustering:
    def __init__(self, initial_n_clusters = 2, n_clusters = 2, max_iter=30, llk=True, xhmm=True,pool = Pool, llk_epsilon = None ,xhmm_epsilon = None, **kwargs):
        self.n_clusters = n_clusters
        self.initial_n_clusters = initial_n_clusters
        self.max_iter = max_iter
        self.kwargs = kwargs.copy()  
        self.llk_kwargs = kwargs.copy()
        self.labels_ = None
        self.alg = None
        self.done_ = False
        self._llklike = llk
        self._llkalg = None
        self.features = None
        self.xhmm = xhmm
        self._pool = pool
        if llk_epsilon is not None:
            self.llk_kwargs['epsilon'] = llk_epsilon
        if xhmm_epsilon is not None:
            self.kwargs['epsilon'] = xhmm_epsilon
        
    def fit_models(self, data, labels = None):
        if labels is None:
            self.labels_ = pd.DataFrame(np.random.randint(self.initial_n_clusters, size=(data[0].shape[0],1)), index = data[0].index)
        else:
            self.labels_ = labels
        if self.xhmm:
            self.alg = XHMMFeatures(pool = self._pool, **self.kwargs)
            self.alg.fit(data, self.labels_)
        if self._llklike:
            self._llkalg = Featurizer_chained(pool = self._pool, **self.llk_kwargs)
            self._llkalg.fit(data, self.labels_) 
        
    def fit(self, data, labels = None, models_fitted = False):
        #assert not (not models_fitted and self.max_iter > 1), "Run misconfigured"
        if labels is None and not models_fitted:
            self.labels_ = pd.DataFrame(np.random.randint(self.initial_n_clusters, size=(data[0].shape[0],1)), index = data[0].index)
        elif not models_fitted:
            self.labels_ =  labels   
        for i in range(self.max_iter):
            if not models_fitted:
                self.fit_models(data, labels) 
            self.features = self.transform(data)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.features)     
            labels_ = pd.DataFrame(kmeans.labels_, index = self.features.index)
            if is_equal_labels(labels_, self.labels_):
                self.done_ = True
                break
            self.labels_ = labels_
        return self

    def transform(self, data):
        if self._llklike:
            features_llk = self._llkalg.transform(data).replace([np.inf, -np.inf], np.nan)
            if not self.xhmm:
                 return features_llk.fillna(0)
            features = self.alg.transform(data)
            features = pd.concat([features, features_llk], axis=1, join="outer").dropna(axis=1, how='all')
        else:
            return self.alg.transform(data)

        return features.dropna(axis=1, how='all').fillna(0)
    
    def fit_transform(self, data, labels = None):
        self.fit(data, labels)
        return self.features
    
    @property
    def pool(self): 
        return self._pool     
    
    @pool.setter
    def pool(self, value):
        self._pool = value
        self._llkalg.pool =value
        self.alg.pool =value

class Featurizer_chained:
    def __init__(self, featurizer=InferredHMMLikelihood, pool = Pool, **kwargs):
        self.pool = pool
        self._all_kwargs = kwargs.copy()
        self._featurizer = featurizer
        self._featurizers = None
        self._all_kwargs['pickle_models'] = True
        
    def fit(self, data, labels):
        data = _check_data(data, labels)
        self._featurizers =  [self._featurizer(**self._all_kwargs) for x in data]
        self.set_pickle_models(False)
        with callwithValidKwargs(self.pool,self._all_kwargs) as executor:
            self._featurizers = list(executor.map(fit_helper, zip(self._featurizers, data, repeat(labels))))
        self._fitted = True
        self.set_pickle_models(True)
        return self
    
    def transform(self, data):     
        self.set_pickle_models(False)
        with callwithValidKwargs(self.pool,self._all_kwargs) as executor:
            all_features = list(executor.map(transform_helper, zip(self._featurizers, data)))
        '''
        all_f = []
        for f_name in all_features:            
            with open(f_name, 'rb') as f:
                dataframe = pickle.load(f)
            all_f.append(dataframe)
        '''
        self.set_pickle_models(True)
        return pd.concat(all_features, axis=1)
            
    def fit_transform(self, data, labels):
        return self.fit(data, labels).transform(data)

    def set_pickle_models(self, flag):
        for f in self._featurizers:
            f.pickle_models = flag


def transform_helper(arg):
    if True:
        debug_mpi("start")
        q = arg[0]
        data = arg[1]
        ret = q.transform(data)
        debug_mpi("done transporm")
        return ret
        pickle_file_name = RANDOM_NAME()
        with open(pickle_file_name, 'wb') as handle:
            pickle.dump(ret, handle)
        debug_mpi("done writing file")
        debug_mpi(str(ret))
        return pickle_file_name
    #except Exception as e:
    #    debug_mpi(str(e))
    #   return -1
