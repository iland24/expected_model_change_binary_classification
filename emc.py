import numpy as np
import pandas as pd
import random
import os

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 

import matplotlib.pyplot as plt

import logging

### Splitting imbalanced data functions ###

def get_minor_label(label):
    """
    label: 1d np array
    label only contains 1 or 0 (binary label)
    """
    n_zero = np.count_nonzero(label==0)
    n_one = np.count_nonzero(label==1)
    if n_zero >= n_one:
        return 1
    elif n_one > n_zero:
        return 0
    else:
        raise Exception('Input label contains more than 2 labels;\
        need to input binary label.')
    
def ratio_or_int(oversample_size):
    """returns """
    if isinstance(oversample_size, float):
        if 0.0<oversample_size<=1.0:
            return 'ratio'
        elif 0.0==oversample_size:
            return 'zero'
    elif isinstance(oversample_size,int):
        if 0<oversample_size:
            return 'int'
        elif 0==oversample_size:
            return 'zero'
    else:
        raise TypeError("'oversample_size' \
        param must be an integer or a float in range [0.0,1.0].")

# Function to oversample minor label to make binary classes balanced in the train sample
def train_test_split_oversample(data, label, sample_ids=None, 
                                oversample_type='ros', split_ratio=0.8, 
                                oversample_size=0.2, seed=0):
    '''
    data(2d np array)
    
    label(1d np array): binary label of data
    
    sample_ids(1d np array): input sample_ids 
    to track samples when samples are selected in EMC;
    if sample_ids!=None, new ids labeled 'oversampled_#' 
    for oversampled data will be added to original input
    and returned;
    
    oversample_type(str): 'ros' (random over sampler) or 
    'smote' (Synthetic Minority Over-sampling Technique)
    
    split_ratio(float(0.0,1.0]): ratio of minor label train data; split base 
    on total number of minor label samples
    
    oversample_size (int or float(0.0,1.0]): minor label train data to 
    be oversampled after number of major label samples are matched 
    to number of minor label samples; 
        if int, designated positive integer of major label will be
        added and minor label train samples will be oversampled;
        
        if ratio, ratio of remaining minor label training data will 
        be oversampled
        (# oversampled minor label data) = (# minor label train samples)*ratio 
    
    seed: simulation number
    
    *Note: this function is designed initially to match the 
    number of minor and major labels in the test data. 
    Number of  test set is determined based on how much the
    user will sample from minor label through split_ratio 
    parameter.
    
    *train/test split is done before oversampling
    to prevent data leakage to the test data.
    '''
    assert (isinstance(split_ratio,float) and 0.0<split_ratio<1.0), \
    "'split_ratio' param must be a float in range (0.0,1.0]"
    
    # randomly permute samples
    np.random.seed(seed+10)
    
    # get minor label count (print minor label count)
    minor_label = get_minor_label(label)
    
    minor_label_idx = np.where(label==minor_label)[0]
    minor_label_cnt = len(minor_label_idx)
    major_label_idx = np.where(label!=minor_label)[0]
    major_label_cnt = len(major_label_idx)
     
    print('Total minor label cnt:', minor_label_cnt)
    print('Total major label cnt:', major_label_cnt, end='\n\n')
    
    # Use split_ratio on total minor label data to determine how many minor label 
    # will be included in test. Same numer of major label will be added to make 1:1 ratio of each label.
    n_minor_label_test = minor_label_cnt - int(minor_label_cnt*split_ratio)
    n_minor_label_train = minor_label_cnt - n_minor_label_test
    print('Test minor label cnt:', n_minor_label_test)
    print('Train minor label cnt before oversampling:', n_minor_label_train)
    
    assert oversample_size < (major_label_cnt-n_minor_label_test), \
    "'oversample_size' param too big. 'oversample_size' should not exceed number of major label train samples"
    
    out = ratio_or_int(oversample_size)
    if out=='ratio':
        n_samples_to_oversample = int(n_minor_label_train*oversample_size)
    elif out=='int':
        n_samples_to_oversample = oversample_size
    else:
        n_samples_to_oversample = 0
    print('Train minor label samples to be oversampled:',n_samples_to_oversample, end='\n\n')
    
    # get test/train minor label and shuffle
    # randomly permutate before splitting to Test&Train
    minor_label_random_idx = np.random.permutation(minor_label_idx)
    minor_label_test_idx = minor_label_random_idx[:n_minor_label_test]
    minor_label_train_idx = minor_label_random_idx[n_minor_label_test:]
    
    # get test/train major label and shuffle
    major_label_idx_random = np.random.permutation(major_label_idx)
    major_label_test_idx = major_label_idx_random[:n_minor_label_test]
    major_label_train_idx = major_label_idx_random[n_minor_label_test:]
    
    # combine major/minor TEST data&label
    test_x = np.concatenate([data[minor_label_test_idx], data[major_label_test_idx]],axis=0)
    test_y = np.concatenate([label[minor_label_test_idx],label[major_label_test_idx]],axis=0)
    assert len(test_x) == len(test_y), "test data and label length doesn't match"
    
    # oversample major label using n_samples_to_oversample
    oversampled_major_label_train_idx = major_label_train_idx[:(n_minor_label_train+n_samples_to_oversample)]
    
    # combine major/minor TRAIN data&label
    train_x = np.concatenate([data[minor_label_train_idx], data[oversampled_major_label_train_idx]],axis=0)
    train_y = np.concatenate([label[minor_label_train_idx], label[oversampled_major_label_train_idx]],axis=0)
    
    assert len(train_x) == len(train_y), "train data and label length doesn't match"
    print(f'Train size before resampling: {len(train_y)}')
    
    if sample_ids is not None:
        test_ids = sample_ids[np.concatenate([minor_label_test_idx, major_label_test_idx])]
        train_ids = sample_ids[np.concatenate([minor_label_train_idx, oversampled_major_label_train_idx])]
    
    # resample TRAINING DATA only
    if oversample_type=='ros':
        ros = RandomOverSampler(sampling_strategy='minority', random_state=seed)
        x_resampled_train, y_resampled_train = ros.fit_resample(train_x, train_y)
    elif oversample_type=='smote':
        smote = SMOTE(random_state=seed)
        x_resampled_train, y_resampled_train = smote.fit_resample(train_x, train_y)
    else:
        raise ValueError("oversample_type param must either be 'ros' or 'smote'")
    
    shuffled_idx = np.random.permutation(len(x_resampled_train))
    x_resampled_train = x_resampled_train[shuffled_idx]
    y_resampled_train = y_resampled_train[shuffled_idx]
    
    print(f'Train size after resampling: {len(y_resampled_train)}')
    print("Test size: ",len(test_x))
    print()
    
    # if oversampled data > 0
    if (sample_ids is not None) and len(y_resampled_train)>=len(train_ids):
        # add new sample ids for oversampled samples
        n_added = len(y_resampled_train)-len(train_ids)
        train_ids = np.concatenate([train_ids,
                                           np.array(['oversampled_'+str(i) for i in range(n_added)])])
        train_ids = train_ids[shuffled_idx]
        
        return x_resampled_train, y_resampled_train, test_x, test_y, test_ids, train_ids
    else:
        return x_resampled_train, y_resampled_train, test_x, test_y


### helper functions ###

def make_unique_file_name(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    return path


### EMC functions ###

class LogR(LogisticRegression):
    '''
    logistic regression class
    '''
    def __init__(self, train_x, train_y, random_state=0, max_iter=500):
        super().__init__(random_state=random_state, max_iter=max_iter)
        self.train_x = train_x
        self.train_y = train_y.reshape(len(train_y),)

    def fit_model(self):
        self.fit(self.train_x, self.train_y)

    def generate_accuracy(self, true_y, pred_y):
        return accuracy_score(true_y, pred_y)
    
    def generate_precision(self, true_y, pred_y):
        return precision_score(true_y, pred_y)
    
    def generate_recall(self, true_y, pred_y):
        return recall_score(true_y, pred_y)
    
def fit_logR_model(data):
    """
    data (2d np array): data & sample id & label
    *second to last righ-most column == sample id
    *last righ-most column == label
    
    returns fitted model
    """
    x = data[:, :len(data[0])-2]
    y = data[:,len(data[0])-1]
    model = LogR(x, y)
    model.fit_model()
    return model

def calc_gradient(model, x, y_hat):
    """
    Calculates and returns "expected" graident of sklearn.linear_model.LogisticRegression
    loss function (cross entropy with l2 regularization).
    
    x: new sample from data_pool; shape=(1, n_feat)
    y: true label; shape=(1,)
    y_hat: p(y=1|x); shape=(1,) (probability)
    w: LogisticRegression's model.coef_ ; shape=(1, n_feat)

    return gradient w.r.t. w (all weights); shape=(1, n_feat)
    
    *p(y=1|x) = 1/(1+e^(Wx))
    """
    exp_mod_change=0.0
    model_classes = model.classes_
    # print(model.classes_)  # => [0,1]
    # w = model.coef_[0]  # w:(1, n_features)
    for m_class in model_classes:
        # Use prob of both labels multiplied by gradient to get the "expected" model change
        exp_mod_change += model.predict_proba([x])[0][int(m_class)]*np.linalg.norm((y_hat-m_class)*x)
        # model.predict_proba([x])[0][int(m_class)] => indexing to get prob of each class (0 or 1)
    return exp_mod_change

def emc(data, data_pool):
    '''
    data: Queried data (last col = label)
    data_pool: data pool from which we select new sample (last col = label)

    Loop through data_pool to look for
    a sample that changes the gradient the most.

    Returns selected sample.
    '''
    max_v = float('-inf')
    max_idx = -1

    #loop through data_pool & search sample that brings biggest change in gradient
    for i, row in enumerate(data_pool):
        new_sample = row[:len(row)-2]
        
        # fit on data temporarily attached new sample
        model = fit_logR_model(np.concatenate([data,[row]],axis=0))
        
        # calc p(y=1|new_x)
        y_hat = model.predict_proba([new_sample])[:,1]
        exp_mod_change = calc_gradient(model, new_sample, y_hat)
        if exp_mod_change > max_v:
            max_v = exp_mod_change
            max_idx = i

    # update remaining data with drop() & return one new sample
    sampled_x = data_pool[max_idx]
    data_pool = np.delete(data_pool, max_idx, axis=0)
    
    return sampled_x, data_pool

def calc_acc_prec_rec(model, test_x, test_y):
    y_pred = model.predict(test_x)
    accuracy = model.generate_accuracy(test_y, y_pred)
    precision = model.generate_precision(test_y, y_pred)
    recall = model.generate_recall(test_y, y_pred)
    return accuracy, precision, recall

def run_l_simulations_emc(data_x, label_y, sample_ids=None, **config):
    '''
    data (2d np array)
    label (1d np array): data label (binary)
    
    Trains logistic regression (LR), adding 1 sample 
    at a time for each update step using 
    expected model change technique.

    Initially train LR with initial_train_ratio of 
    training data.
    
    Returns n_sim by m np.arrays (acc/prec/recall)
    (1) n_sim number of simulations
    (2) m rounds of updates
    '''
    assert len(data_x) == len(label_y)
    
    n_sim = config['n_sim']
    initial_train_ratio = config['initial_train_ratio']
    log_freq = config['log_freq']
    emc_dir_path = config['emc_dir_path']
    
    # Simulation loop
    for i in range(n_sim):
        logger_emc.info(f'EMC simulation round: {i}\n')
        print(f'EMC simulation round: {i}\n')
        
        # Oversample
        train_x, train_y, test_x, test_y, test_ids, train_ids = \
            train_test_split_oversample(data=data_x, label=label_y, 
                                        sample_ids=sample_ids,
                                        oversample_type=config['oversample_type'],
                                        split_ratio=config['split_ratio'],
                                        oversample_size=config['oversample_size'],
                                        seed=i)
        train_y = train_y.reshape(-1,1)
        test_y = test_y.reshape(-1,1)
        train_ids_idx = np.arange(len(train_ids)).reshape(-1,1)
        if sample_ids is not None:
            np.save(f'./{emc_dir_path}/test_sample_ids.npy', test_ids)
        
        logger_emc.info(f'train data shape: {train_x.shape}')
        logger_emc.info(f'test data shape: {test_x.shape}\n')

        # initial random sampling
        init_size = round(len(train_x) * initial_train_ratio)
        assert init_size >= 10,\
        "Initial training size less than 10. Increase initial_train_ratio param \
        or increase number of samples." 
        
        n_updates = len(train_x)-init_size

        # attach sample id & label
        data_w_label = np.concatenate([train_x, train_ids_idx, train_y], axis=1)
        
        # divide into data (for fitting model) and data_pool (for selecting new sample)
        data = data_w_label[0:init_size]
        data_pool = data_w_label[init_size:]
        
        logger_emc.info(f'initial data shape: {data.shape}')
        logger_emc.info(f'initial data_pool shape: {data_pool.shape}\n')
        
        # store accuracy / precision / recall sums for each update to get avg of folds
        ith_sim_accuracy_updates_ls = []
        ith_sim_precision_updates_ls = []
        ith_sim_recall_updates_ls = []
        ith_sim_id_update_order_ls = []
        
        ###### Performance of LR trained with inital train data ######
        first_model=fit_logR_model(data)
        accuracy, precision, recall = calc_acc_prec_rec(first_model, test_x, test_y)
        ith_sim_accuracy_updates_ls.append(accuracy)
        ith_sim_precision_updates_ls.append(precision)
        ith_sim_recall_updates_ls.append(recall)
        logger_emc.info(f'\n initial accuracy: {accuracy}\n initial precision: {precision}\n initial recall: {recall}\n')
        ##############################################################
        
        # update n times, choosing one sample at each loop from data_pool
        for j in range(n_updates):
            # sample one using expected model change query.
            sampled_x, data_pool = emc(data, data_pool)
            if sample_ids is not None:
                # log id of the selected sample
                logger_emc.info(f'Selected sample id: {train_ids[sampled_x[len(sampled_x)-2]]}')
                ith_sim_id_update_order_ls.append(train_ids[sampled_x[len(sampled_x)-2]])
            
            # add selected sample to data
            data = np.concatenate([data, [sampled_x]], axis=0)
            
            # fit new model and measure performance on test set
            updated_model=fit_logR_model(data)
            accuracy, precision, recall = calc_acc_prec_rec(updated_model, test_x, test_y)

            #append curr update round, and curr accuracy as tuple
            ith_sim_accuracy_updates_ls.append(accuracy)
            ith_sim_precision_updates_ls.append(precision)
            ith_sim_recall_updates_ls.append(recall)
            
            if j%log_freq==0:
                logger_emc.info(f'''{j}th update\n\
                data shape: {data.shape}\n\
                data_pool shape: {data_pool.shape}\n\
                accuracy: {accuracy}\n\
                precision: {precision}\n\
                recall: {recall}\n\n''')

        # Save metrics inside simulation loop in case loop gets interrupted 
        np.save(f'./{emc_dir_path}/{i}_sim_emc_accuracy.npy', ith_sim_accuracy_updates_ls)
        np.save(f'./{emc_dir_path}/{i}_sim_emc_precision.npy', ith_sim_precision_updates_ls)
        np.save(f'./{emc_dir_path}/{i}_sim_emc_recall.npy', ith_sim_recall_updates_ls)
        if sample_ids is not None:
            np.save(f'./{emc_dir_path}/{i}_sim_emc_sample_ids.npy', ith_sim_id_update_order_ls)

### Random sampling functions ###

def random_selection_of_sample(data, data_pool, seed):
    '''
    data: queried data (last col = label)
    data_pool: data pool from which we select new sample (last col = label)

    Select random sample from data_pool.
    
    Returns selected sample.
    '''
    # rand generator seed is same in each simulation
    np.random.seed(seed) 
    rand_idx = np.random.randint(0, data_pool.shape[0])
    sampled_x = data_pool[rand_idx]
    data_pool = np.delete(data_pool, rand_idx, axis=0)
    
    return sampled_x, data_pool

def run_n_simulations_random_sampling(data_x, label_y, **config):
    '''
    data (2d np array)
    label: data label (binary)
    
    Trains logistic regression (LR), adding 1 sample 
    randomly at a time for each update step.
    
    Initially train LR with initial_train_ratio of 
    training data.
    
    Returns n_sim by m np.arrays (acc/prec/recall)
    (1) n_sim number of simulations
    (2) m rounds of updates
    '''
    assert len(data_x) == len(label_y)
    
    n_sim = config['n_sim']
    initial_train_ratio = config['initial_train_ratio']
    log_freq = config['log_freq']
    rand_sampling_dir_path = config['rand_sampling_dir_path']
    # Simulation loop
    for i in range(n_sim):
        logger_rand_sampler.info(f'Random sampling simulation round: {i}\n')
        print(f'Random sampling simulation round: {i}\n')
        
        # Oversample
        train_x, train_y, test_x, test_y = \
            train_test_split_oversample(data=data_x, label=label_y, 
                                        oversample_type=config['oversample_type'],
                                        split_ratio=config['split_ratio'],
                                        oversample_size=config['oversample_size'], 
                                        seed=i)
        
        train_y = train_y.reshape(-1,1)
        test_y = test_y.reshape(-1,1)
        logger_rand_sampler.info(f'train data shape: {train_x.shape}')
        logger_rand_sampler.info(f'test data shape: {test_x.shape}\n')

        # initial random sampling
        init_size = round(len(train_x) * initial_train_ratio)

        n_updates = len(train_x)-init_size

        # attach label
        data_w_label = np.concatenate([train_x, train_y], axis=1)
        
        # divide into data (for fitting model) and data_pool (for selecting new sample)
        data = data_w_label[0:init_size]
        data_pool = data_w_label[init_size:len(data_w_label)]

        logger_rand_sampler.info(f'initial data shape: {data.shape}')
        logger_rand_sampler.info(f'initial data_pool shape: {data_pool.shape}\n')
        
        # store accuracy / precision / recall sums for each update to get avg of folds
        ith_sim_accuracy_updates_ls = []
        ith_sim_precision_updates_ls = []
        ith_sim_recall_updates_ls = []
        
        ###### Performance of LR trained with inital train data ######
        first_model=fit_logR_model(data)
        accuracy, precision, recall = calc_acc_prec_rec(first_model, test_x, test_y)
        ith_sim_accuracy_updates_ls.append(accuracy)
        ith_sim_precision_updates_ls.append(precision)
        ith_sim_recall_updates_ls.append(recall)
        logger_rand_sampler.info(f'\n initial accuracy: {accuracy}\n initial precision: {precision}\n initial recall: {recall}\n')
        ##############################################################
        
        # update n times, choosing one sample at each loop from data_pool
        for j in range(n_updates):
            # sample one using random query
            sampled_x, data_pool = random_selection_of_sample(data, data_pool, seed=i)
            # add selected sample to data
            data = np.concatenate([data, [sampled_x]], axis=0)
            
            # fit new model and measure performance on test set
            updated_model=fit_logR_model(data)
            accuracy, precision, recall = calc_acc_prec_rec(updated_model, test_x, test_y)

            #append curr update round, and curr accuracy as tuple
            ith_sim_accuracy_updates_ls.append(accuracy)
            ith_sim_precision_updates_ls.append(precision)
            ith_sim_recall_updates_ls.append(recall)
            
            if j%log_freq==0:
                logger_rand_sampler.info(f'''{j}th update\n\
                            data shape: {data.shape}\n\
                            data_pool shape: {data_pool.shape}\n\
                            accuracy: {accuracy}\n\
                            precision: {precision}\n\
                            recall: {recall}\n\n''')

        # Save metrics inside simulation loop in case loop gets interrupted 
        np.save(f'./{rand_sampling_dir_path}/{i}_sim_rand_sample_accuracy.npy',ith_sim_accuracy_updates_ls)
        np.save(f'./{rand_sampling_dir_path}/{i}_sim_rand_sample_precision.npy', ith_sim_precision_updates_ls)
        np.save(f'./{rand_sampling_dir_path}/{i}_sim_rand_sample_recall.npy', ith_sim_recall_updates_ls)

### Metric plotting functions ###

def plot_metrics(perf_metric_ls, n_updates_to_plot, plot_separate_sim=False, **kwargs):
    """
    plot performance metric
    
    n_updates_to_plot: number of updates to plot (x axis)
    plot_separate_sim: if False, plot average performance, else plot separate lines
    """
    # x => update steps
    x = np.linspace(1, n_updates_to_plot, n_updates_to_plot)
        
    if plot_separate_sim:
        # Generate n_sim separate perf_metric lines
        for i in range(len(perf_metric_ls)):
            plt.plot(x, perf_metric_ls[i][:n_updates_to_plot], color=kwargs['color'])
            
    else:
        # Generate average perf_metric at each update
        avg_pef_metric = np.sum(perf_metric_ls,axis=0) / len(perf_metric_ls)
        plt.plot(x, avg_pef_metric[:n_updates_to_plot], **kwargs)
        plt.legend(loc='lower right')
        