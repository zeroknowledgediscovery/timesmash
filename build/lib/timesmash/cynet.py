import numpy as np
import pandas as pd
import subprocess
import os
import time
from itertools import repeat, product, chain
import glob 
import shutil 
from timesmash import Quantizer
from functools import partial
import traceback
from timesmash.utils import BIN_PATH, RANDOM_NAME, callwithValidKwargs
from multiprocessing import Pool
from sklearn import metrics
import time
bin_path = BIN_PATH
CYNET_PATH = bin_path + 'cynet'
FLEXROC_PATH = bin_path + 'flexroc'
XgenESeSS_PATH = bin_path + 'XgenESeSS_cynet'

import time

ERROR_RIDIRECT = False
DEBUG_MPI = False
MPI_DEBUG_OUTPUT_FOLDER = './debug_output'

def debug_mpi(*message, sleep = False):
    out_message = ''
    for string in message:
        out_message = out_message + str(string)
    if DEBUG_MPI or (ERROR_RIDIRECT and ('rror:' in out_message)):
        if not os.path.isdir(MPI_DEBUG_OUTPUT_FOLDER):
            os.mkdir(MPI_DEBUG_OUTPUT_FOLDER)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank() 
        if sleep:
            time.sleep(int(rank))
        file = '{}/rank_{}.txt'.format(MPI_DEBUG_OUTPUT_FOLDER, rank)
        if os.path.exists(file):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        with open(file,append_write) as f:
            f.write(str(out_message)+ '\n')

class XHMMFeatures:
    def __init__(self,*, pool = Pool, **kwargs):
        self._quantizer = None
        self._discreet_feturizer = []
        self._fitted = False
        self._pool = pool
        self._all_kwargs = kwargs
        self._pickle_models = True
        #assert 'detrending' in self._all_kwargs

    def fit(self, data, labels):
        debug_mpi('enter fit')
        data = _check_data(data, labels)
        self._quantizer = Quantizers_chained(pool =  self._pool,**self._all_kwargs)
        q_iter = self._quantizer.fit_transform(data, labels)
        features = []
        for q in q_iter:
            featurizer = _AUC_Feature(pool =  self._pool,**self._all_kwargs)
            featurizer._set_up(q, labels)
            self._discreet_feturizer.append(featurizer)
        debug_mpi("setup done")
        iters = chain.from_iterable([it._get_iter() for it in self._discreet_feturizer])
        debug_mpi("start fitting models")
        with callwithValidKwargs(self._pool,self._all_kwargs) as executor:
            executor.map(run_XgenESeSS, iters)
        debug_mpi("done fitting models")
        self._fitted = True
        return self

    
    def transform(self, data):
        q_iter = self._quantizer.transform(data)
        feature = []
        for q, featurizer in zip(q_iter, self._discreet_feturizer):
            features = featurizer.transform(q)
            feature.append(features)
        res = pd.concat(feature, axis=1, sort=True)
        res = res.reindex(sorted(res.columns), axis=1)
        res = res.loc[data[0].index]

        return res
    
    def fit_transform(self, data, labels):
        return self.fit(data, labels).transform(data)

    @property
    def pool(self): 
        return self._pool     
    
    @pool.setter
    def pool(self, value):
        self._pool = value
        for f in self._discreet_feturizer:
            f.pool = value
        self._quantizer.pool = value

    @property
    def pickle_models(self): 
        return self._pickle_models     
    
    @pickle_models.setter
    def pickle_models(self, value):
        self._pickle_models = value
        for f in self._discreet_feturizer:
            f._pickle_models = value
    
def fit_helper(arg):
    debug_mpi("fit start")

    q = arg[0]
    data = arg[1]
    label = arg[2]
    q.fit(data, label)
    debug_mpi("fit end")

    return q

    
class Quantizers_chained:
    def __init__(self, return_failed=True, n_quantizations = 10, pool = Pool, max_alphabet_size=2, detrending=0, **kwargs ):
        self.pool = pool
        self._all_kwargs = kwargs
        self._quantizers = None
        self._all_kwargs['return_failed'] = return_failed
        self._all_kwargs['detrending'] = detrending
        self._all_kwargs['n_quantizations'] = n_quantizations
        self._all_kwargs['prune'] = False
        self._all_kwargs['max_alphabet_size'] = max_alphabet_size
   
 
    def fit(self, data, labels):
        data = _check_data(data, labels)
        #print('fitting', len(data))

        with callwithValidKwargs(self.pool,self._all_kwargs) as executor:
            self._quantizers = list(executor.map(fit_helper, zip([callwithValidKwargs(Quantizer, self._all_kwargs) for x in data], data, repeat(labels))))
        self._fitted = True
        return self

    def _fit_iters(self, data, labels):
        for df in data:
            qtz = Quantizer(**self._all_kwargs)
            self._quantizers.append(qtz)
            yield partial(qtz.fit, data, labels) 
    
    def transform(self, data):
        zipped = zip(*[self._quantizers[i].transform(x) for i,x in enumerate(data)])
        for dfs in zipped:
            yield list(dfs)
    
    def fit_transform(self, data, labels):
        return self.fit(data, labels).transform(data)


def run_XgenESeSS(to_call):
    try:
        debug_mpi('start generating models')
        out = subprocess.check_output(to_call, shell=True)  
        debug_mpi('end generating models')

        return out
    except Exception as e:
        debug_mpi('error xgen: ' + str(e) + str(traceback.format_exc()))
        raise e
    
def get_auc(args):
    try:
    #if True:
        debug_mpi('in mpi: '+str(args) + ' end arg', sleep = True)
        data = args[0][0].dropna(axis=1)
        person = args[0][1]
        models = args[1]
        kwargs = args[2]
        folder_name = RANDOM_NAME()
        debug_mpi('dir making:', folder_name)
        os.mkdir(folder_name)
        debug_mpi('dir made :', folder_name,  os.path.isdir(folder_name))
        assert( os.path.isdir(folder_name))
        for ts in data.index:
            temp_df = pd.DataFrame(data.loc[ts]).T
            temp_df.to_csv('{}/{}'.format(folder_name,ts),index=False, header=False, sep =' ')
            assert(os.path.isfile('{}/{}'.format(folder_name,ts)))
            debug_mpi('writing ts: {}   {}/{}'.format(ts,folder_name,ts), os.path.isfile('{}/{}'.format(folder_name,ts)))
        res = []
        debug_mpi('starting cynet:' + str(data.index))
        for ts in data.index:
            debug_mpi('get_auc for loop: ' + str(ts))

            stored_model = models[ts]
            auc, _, _ = _simulateModel(MODEL_PATH = stored_model, DATA_PATH = folder_name+'/', RUNLEN = data.shape[1], **kwargs).run(**kwargs)
            res.append(auc)
            debug_mpi('get_auc for loop end: ' +  str(ts))
        shutil.rmtree(folder_name)
        if 'get_log_file' in kwargs and kwargs['get_log_file']:
            return res
        df = pd.DataFrame(res).T
        df.columns = [models[name] for name in models]
        df.index = [person]
        debug_mpi('get_auc return', str(df))
        return df
    
    except Exception as e:
        debug_mpi('error message3: ' + str(e) + str(traceback.format_exc()))
        raise e
    
class _AUC_Feature:
    def __init__(self,*, pool = Pool, **kwargs ):
        self._model_manager = {}
        self._fitted = False
        self._fitted = False
        self.pool = pool
        self._all_kwargs = kwargs
        self.pickle_models = True
        self._it_list = []
        
    def _set_up(self, data, labels):
        debug_mpi("enter setup")
        data = _check_data(data, labels)
        for label in labels[labels.columns[0]].unique():
            flat_data =  _flatten_data(data, labels, label)
            self._model_manager[label] = callwithValidKwargs(_xgModels ,self._all_kwargs)._setup_files(flat_data)
            self._it_list.append(self._model_manager[label]._get_commands())
        debug_mpi("exit setup")
        return self

    def _run(self):
        with callwithValidKwargs(self.pool,self._all_kwargs) as executor:
            it = self._get_iter()
            [executor.map(run_XgenESeSS, it)]
        '''
        for label in labels[labels.columns[0]].unique():
            print(label)
            print(self._model_manager[label]._models)'''
        self._fitted = True
        return self
    
    def _get_iter(self):
        self._fitted = True
        it = chain.from_iterable(self._it_list)
        self._it_list = None
        return it

    def fit(self, data, labels):
        self._set_up(data, labels)._run()
        return self
    
    def transform(self, data):

        it = _list_of_dataframe_iter(data)
        debug_mpi("transform _AUC_Feature 1")
        with callwithValidKwargs(self.pool,self._all_kwargs) as executor:
        #with self.pool(unordered = True) as executor:

            args = product(it, [self._model_manager[name]._models for name in self._model_manager],[self._all_kwargs])
            debug_mpi("transform before execute")

            res = executor.map(get_auc, args)
            debug_mpi("transform _AUC_Feature 2")
        if 'get_log_file' in self._all_kwargs and self._all_kwargs['get_log_file']:
            return list(res)
        
        features = pd.concat(res, sort=False, axis = 0, join = 'outer' )
        debug_mpi("transform _AUC_Feature 3")

        features = features.reset_index().groupby('index').max()
        features = features.sort_index(axis=1)
        return features
    
    def fit_transform(self, data, labels):
        return self.fit(data, labels).transform(data)
    
    @property
    def pickle_models(self): 
        return self._pickle_models     
    
    @pickle_models.setter
    def pickle_models(self, value):
        self._pickle_models = value
        for label in self._model_manager:
            self._model_manager[label] = value
            
def _check_data(data, labels):
    index = labels.index
    
    return data

def _list_of_dataframe_iter(data):
    indexes = data[0].index
    for person in indexes:
        person_data =  pd.DataFrame()
        for i, channel in enumerate(data):
            person_data[i] = channel.loc[person]
        person_data = person_data.T
        yield person_data, person

def _flatten_data(data, labels, label):
    labels = labels[labels[labels.columns[0]] == label]
    index = labels.index
    flat_data = pd.DataFrame()
    for i, df in enumerate(data):
        flat_data[i] = df.loc[index].values.flatten()
    return flat_data.T

class _xgModels:
    '''
    Utility class for running XgenESeSS. This class will either run XgenESeSS
    locally or produce the list of commands to run on a cluster. We note that
    you may set the path of XgenESeSS in the yaml file. If running on a cluster
    then the commands will use the path use the XgenESeSS path in the yaml. If
    running on
    Attributes -
        TS_PATH(string)- path to file which has the rowwise multiline
            time series data
        NAME_PATH(string)-path to file with name of the variables
        LOG_PATH(string)-path to log file for xgenesess inference
        BEG(int) & END(int)- xgenesses run parameters (not hyperparameters,
            Beg is 0, End is whatever tempral memory is)
        NUM(int)-number of restarts (20 is good)
        PARTITION(float)-partition sequence
        XgenESeSS_PATH(str)-path to XgenESeSS
        RUN_LOCAL(bool)- whether to run XgenESeSS locally or produce a list of
        commands to run on a cluster.
    '''
    def __init__(self,
                 *,
                 delay_min=0,
                 delay_max=0,
                 NUM=40,
                 DERIVATIVE=0,
                 self_models = False,
                 epsilon = 0.025,
                 CAP_P=False):

        debug_mpi(epsilon)
        self.TS_PATH = RANDOM_NAME(clean=True)
        self.NAME_PATH = RANDOM_NAME(clean=True)
        self.FILEPATH = RANDOM_NAME(clean=True)
        self.BEG = delay_min
        self.END = delay_max
        self.NUM = NUM
        self.DERIVATIVE = DERIVATIVE
        self.epsilon = epsilon
        self.pickle_models = True
        self._models = {}
        self._commands = []
        self._fitted = True
        self._indexes = True
        if self_models:
            self._self_flag = ' -S '
        else:
            self._self_flag = ''

    def _get_commands(self):
        for INDEX, name in enumerate(self._indexes):
            model_file = RANDOM_NAME(clean=True)
            #print(model_file)
            self._models[name] = model_file + 'model.json'
            xgstr = XgenESeSS_PATH +' -f ' + self.TS_PATH\
                 + " -k \"  :" + str(INDEX) +  " \"  -B " + str(self.BEG)\
                 + "  -E " +str(self.END) + ' -n ' +str(self.NUM)\
                 + ' -T symbolic' + ' -N '\
                 + self.NAME_PATH + ' -u '+ str(self.DERIVATIVE) +' -m -G 10000 -v 0 -A 1 -q -w '+  model_file +  self._self_flag + ' -g 0.01' + ' -e ' + str(self.epsilon)
            yield xgstr


    def _setup_files(self, data):
        # change based on python version
        cols = pd.DataFrame(data.columns)
        idx = pd.DataFrame(data.index)
        if idx.shape[1] == 1: # test for pandas compatibility 
            cols = cols.T
            idx = idx.T

        cols.to_csv(self.FILEPATH, header=False, index=False, sep= '\n', line_terminator = '')
        idx.to_csv(self.NAME_PATH, header=False, index=False, sep= '\n', line_terminator = '')
        data.to_csv(self.TS_PATH, header=False, index=False, sep = ' ')
        self._indexes = data.index
        return self
            
    def get_model_file(self, variable_name):
        file = self._models[variable_name]
        file_exist = os.path.exists(file)
    
        if not os.path.exists(file) and not self._fitted:
            raise RuntimeError('Model files do not exit. Execute the calls returned by _get_commands method.')
        assert file_exist

    def __getstate__(self):
        picked_data = {}
        picked_data['class'] = self.__dict__
        if self.pickle_models:
            picked_data['files'] = {}
            for names, file in self._models.items():
                with open(file,"r") as f:
                    picked_data['files'][names] = f.read()
        return picked_data 
    
    
    def __setstate__(self, d):
        self.__dict__ = d['class']
        if self.pickle_models:
            for name, data in d['files'].items():

                file_name = RANDOM_NAME(clean=True)
                #self._models[name] = name
                with open(self._models[name],"w") as f:
                    f.write(data)

                    
class _simulateModel:
    '''
    Use the subprocess library to call cynet on a model to process
    it and then run flexroc on it to obtain statistics: auc, tpr, fuc.
    Input -
        MODEL_PATH(string)- The path to the model being processed.
        DATA_PATH(string)- Path to the split file.
        RUNLEN(integer)- Length of the run.
        READLEN(integer)- Length of split data to read from begining
        CYNET_PATH - path to cynet binary.
        FLEXROC_PATH - path to flexroc binary.
    '''

    def __init__(self,*,
                 MODEL_PATH,
                 DATA_PATH,
                 RUNLEN,
                 READLEN=None,
                 DERIVATIVE=0,
                 CAP_P = False,
                 use_flex_roc = False,
                 **kwargs):
        assert os.path.exists(CYNET_PATH), "cynet binary cannot be found."
        assert os.path.exists(FLEXROC_PATH), "roc binary cannot be found."
        assert os.path.exists(MODEL_PATH), "model file cannot be found: " + MODEL_PATH
        assert any(glob.iglob(DATA_PATH+"*")), \
            "split data files cannot be found."
        debug_mpi(' in sim models __init__')
        self.MODEL_PATH = MODEL_PATH
        self.DATA_PATH = DATA_PATH
        self.RUNLEN = RUNLEN
        self.CYNET_PATH = CYNET_PATH
        self.FLEXROC_PATH = FLEXROC_PATH
        self.RUNLEN = RUNLEN
        self.DERIVATIVE = DERIVATIVE
        self._use_flex_roc = use_flex_roc
        if CAP_P:
            self.p = ' -P '
        else:
            self.p = ' -p '

        if READLEN is None:
            self.READLEN = RUNLEN
        else:
            self.READLEN = READLEN

    def run(self,*,
            LOG_PATH=None,
            #PARTITION=0.5,
            DATA_TYPE='symbolic',
            FLEXWIDTH=1,
            FLEX_TAIL_LEN=-1,
            POSITIVE_CLASS_COLUMN=5,
            EVENTCOL=3,
            tpr_threshold=0.85,
            fpr_threshold=0.15,
            get_log_file = False,
                 **kwargs):

        '''
        This function is intended to replace the cynrun.sh shell script. This
        function will use the subprocess library to call cynet on a model to process
        it and then run flexroc on it to obtain statistics: auc, tpr, fuc.
        Input -
           LOG_PATH (string)- Logfile from cynet run
           PARTITION (string)- Partition to use on split data
           FLEXWIDTH (int)-  Parameter to specify flex in flwxroc
           FLEX_TAIL_LEN (int)- tail length of input file to consider [0: all]
           POSITIVE_CLASS_COLUMN (int)- positive class column
           EVENTCOL (int)- event column
           tpr_threshold (float)- minimum tpr threshold
           fpr_threshold (float)- maximum fpr threshold

        Output -
            auc (float)- Area under the curve
            tpr (float)- True positive rate at specified maximum false positive rate
            fpr (float)- False positive rate at specified minimum true positive rate
        '''
        if FLEX_TAIL_LEN == -1:
            FLEX_TAIL_LEN = self.RUNLEN
            total_run = self.RUNLEN
        if get_log_file:
            total_run = self.RUNLEN + FLEX_TAIL_LEN
        if LOG_PATH is None:
            LOG_PATH = RANDOM_NAME(clean=True)
        cyrstr = self.CYNET_PATH + ' -J ' + self.MODEL_PATH\
            + ' -T ' + DATA_TYPE + ' -N '\
            + str(total_run) + ' -x ' + str(self.READLEN)\
            + ' -l ' + LOG_PATH\
            + ' -w ' + self.DATA_PATH + ' -U ' + str(self.DERIVATIVE) + ' -H 0'
        debug_mpi('start: ' + cyrstr)
        output = subprocess.check_output(cyrstr, shell=True)
        debug_mpi('end: ' + cyrstr)
        if get_log_file:
            return LOG_PATH, None, None
        if not self._use_flex_roc:
            try:
                log_data = pd.read_csv(LOG_PATH, sep ='\s+', header = None)
                log_data_len = log_data.shape[0]
                #log_data = log_data.iloc[int(log_data_len/2):log_data_len]

                os.remove(LOG_PATH)
            except Exception as e:
                os.remove(LOG_PATH)
                #debug_mpi('exit5: ' + cyrstr + str(e) + str(traceback.format_exc()))
                return (np.nan, np.nan, np.nan)
            multi_class = log_data.shape[1]>6
            try:
                if multi_class:
                    auc_new = metrics.roc_auc_score(log_data[3], log_data.loc[:,4:], multi_class='ovr')
                else:
                    auc_new = metrics.roc_auc_score(log_data[3], log_data.loc[:,5])
            except Exception as e:
                if log_data[3].unique().shape[0]==1:
                    #debug_mpi('exit1: ' + cyrstr+ str(e)+ str(traceback.format_exc()))
                    return 1,1,1
                #debug_mpi('exit2: ' + cyrstr+ str(e)+ str(traceback.format_exc()))
                return np.nan, np.nan, np.nan
            debug_mpi('exit3: ' + cyrstr)
            return auc_new, 0, 0
        flexroc_str = self.FLEXROC_PATH + ' -i ' + LOG_PATH\
            + ' -w ' + str(FLEXWIDTH) + ' -x '\
            + str(FLEX_TAIL_LEN) + ' -C '\
            + str(POSITIVE_CLASS_COLUMN) + ' -E ' + str(EVENTCOL)\
            + ' -t ' + str(tpr_threshold) + ' -f ' + str(fpr_threshold)
        output_str = subprocess.check_output(flexroc_str, shell=True)
        results = output_str.split()
        auc = float(results[1])
        tpr = float(results[7])
        fpr = float(results[13])

        debug_mpi('exit4: ' + cyrstr)
        return auc, tpr, fpr

