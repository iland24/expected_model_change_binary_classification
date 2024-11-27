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
import pdb

### Splitting imbalanced data functions ###

def test_val_train_split_w_oversampling(data, label, test_ratio=0.2, val_ratio=None, sample_id=None, 
                                        oversample_type=None, train_minor_label_oversample_ratio=0.2, seed=0):
    """
    if val_ratio=None and sample_id=None,
        returns list(test data, test label, train data, train label)
        
    if val_ratio=None and sample_id!=None,
        returns list(test data, test label, test id, train data, train label, train id)
        
    if val_ratio=None and sample_id=None,
        returns list(test data, test label, validation data, validation label, train data, train label)
    
    if val_ratio=None and sample_id=None,
        returns list(test data, test label, test id, validation data, validation label, validation id, train data, train label, train id)

    data (pandas DataFrame / numpy 2D array)
    label (pandas Series / numpy 1D array): binary (0/1) label 
    oversample_type (str): 'ros' (random over sampler) or 'smote' (Synthetic Minority Over-sampling Technique);
                            if different value is inputted, minor label train data is not oversampled
    sample_id (pandas Series / numpy 1D array): for monitoring which samples belong to which split
    test_ratio (int / float): for setting test split size; can be integer(0<) or ratio ((0:1])
    val_ratio (int / / None float): for setting ; 
                                      if None, no validation split is returned validation split size; can be integer(0<) or ratio ((0:1])
    train_minor_label_oversample_ratio (int / float / None): for setting number of minor label samples to oversample; 
                                                             can be integer(0<) or ratio ((0:1])
    seed: numpy random number generator seed; for shuffling data
    
    Notes:
    *major & minor label sample ratio is kept same in test / validation / train (before oversampling) 
        as the original input data's major & minor label sample ratio
    *this function is similar to sklearn's train_test_split() function; works with binary label (0/1) dataset.
    *train_minor_label_oversample_ratio is used to oversample minor label data in train set
        number of minor label to oversample is calculated by:
            => number_of_minor_label_in_train*train_minor_label_oversample_ratio
    *oversampling can only be done on training set
    
    *best to input tabular data with same type when using pandas DataFrame
    """
    def get_minor_label():
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
    
    def ratio_or_int(num):
        """
        returns 'int', 'ratio' or 'zero'
        if conditions unmet, raises TypeError
        """
        if isinstance(num, float):
            if 0.0<num<=1.0:
                return 'ratio'
            elif 0.0==num:
                return 'zero'
        elif isinstance(num, int):
            if 0<num:
                return 'int'
            elif 0==num:
                return 'zero'
        else:
            raise TypeError("'train_minor_label_oversample_ratio' \
            param must be an integer or a float in range [0.0,1.0].")
    
    def get_split_cnt():
        """
        returns 
        tuple of test / val / train sample count 
        or 
        tuple of test / None / train sample count
        """
        # Get number of test data
        if ratio_or_int(test_ratio)=="ratio":
            n_test = int(n_sample*test_ratio)
        elif ratio_or_int(test_ratio)=="int":
            n_test = test_ratio
        else: # "zero" returned
            raise Exception("0 test data sampled. 'test_ratio' param must be bigger than 0.")
        
        # Check test split size
        if n_test >= n_sample:
            raise Exception("Test size >= total number of samples. Please adjust 'test_ratio' param.")
            
        # Get number of val data
        if val_ratio is None:
            return (n_test, None, n_sample-n_test)
        else:
            if ratio_or_int(val_ratio)=="ratio":
                n_val = int(n_sample*val_ratio)
            elif ratio_or_int(val_ratio)=="int":
                n_val = val_ratio
            else: # "zero" returned
                raise Exception("0 detected for 'val_ratio' param. To exclude validation split, please set 'val_ratio' param as None.")
            
            # Check test/validation split size
            if val_ratio+n_test>=n_sample:
                raise Exception("(Test+Validation) size >= total number of samples. Please adjust 'test_ratio' and/or 'val_ratio' param.")
            return (n_test, n_val, n_sample-n_test-n_val)
    
    def get_split_idx():
        """
        returns 
        tuple of test / val / train sample idx 
        or 
        tuple of test / None / train sample idx
        """
        # Get minor & major label data cnt in each split
        n_minor_test = int(minor_proportion*split_cnt_tup[0])
        n_major_test = split_cnt_tup[0] - n_minor_test

        # final test idx
        minor_test_idx = minor_label_idx[:n_minor_test]
        major_test_idx = major_label_idx[:n_major_test]

        if val_ratio is None:
            # final train idx
            minor_train_idx = minor_label_idx[n_minor_test:]
            major_train_idx = major_label_idx[n_major_test:]

            test_idx = np.concatenate([minor_test_idx, major_test_idx])
            train_idx = np.concatenate([minor_train_idx,major_train_idx])
            return (test_idx, None, train_idx)
        
        else:
            n_minor_val = int(minor_proportion*split_cnt_tup[1])
            n_major_val = split_cnt_tup[1] - n_minor_val
            # final val idx
            minor_val_idx = minor_label_idx[n_minor_test:(n_minor_test+n_minor_val)]
            major_val_idx = major_label_idx[n_major_test:(n_major_test+n_major_val)]
            # final train idx
            minor_train_idx = minor_label_idx[(n_minor_test+n_minor_val):]
            major_train_idx = major_label_idx[(n_major_test+n_major_val):]
            
            test_idx = np.concatenate([minor_test_idx, major_test_idx])
            val_idx = np.concatenate([minor_val_idx, major_val_idx])
            train_idx = np.concatenate([minor_train_idx,major_train_idx])
            return (test_idx, val_idx, train_idx)
            
    # randomly permute samples
    np.random.seed(seed)
    data_is_df=None
    label_is_sr=None
    sample_id_is_sr=None
    if isinstance(data, pd.DataFrame):
        df_columns=data.columns
        data = data.to_numpy()
        data_is_df=True
    if isinstance(label, pd.Series):
        label = label.to_numpy()
        label_is_sr=True
    if isinstance(sample_id, pd.Series):
        sample_id = sample_id.to_numpy()
        sample_id_is_sr=True
        
    n_sample=len(label)

    # get minor label count (print minor label count)
    minor_label = get_minor_label()

    # Get minor and major label index
    minor_label_idx = np.where(label==minor_label)[0]
    major_label_idx = np.where(label!=minor_label)[0]
    
    # Shuffle minor&major label sample indices before sampling
    np.random.shuffle(minor_label_idx)
    np.random.shuffle(major_label_idx)
    
    minor_proportion = len(minor_label_idx)/n_sample
    print(f"Input data's major & minor label ratio:\n{len(major_label_idx)} : {len(minor_label_idx)} = {round(len(major_label_idx)/n_sample*100,2)}% : {round(minor_proportion*100,2)}%",end="\n\n")

    # Get each split's sample cnt (n_test, n_val, n_train)
    split_cnt_tup = get_split_cnt()
    # Get each split's sample idx (test_idx, val_idx, train_idx)
    split_idx_tup = get_split_idx()

    # Again, shuffle combined minor&major label indices before sampling
    # Then, sample data
    np.random.shuffle(split_idx_tup[0])
    test_data = data[split_idx_tup[0]] # test set
    test_label = label[split_idx_tup[0]]
    np.random.shuffle(split_idx_tup[2])
    train_data = data[split_idx_tup[2]] # train set
    train_label = label[split_idx_tup[2]]
    
    if val_ratio is not None:
        np.random.shuffle(split_idx_tup[1])
        val_data = data[split_idx_tup[1]] # val set
        val_label = label[split_idx_tup[1]]
        print(f"Test : Validation : Train\n{len(test_label)} : {len(val_label)} : {len(train_label)}",end="\n\n")
    else:
        print(f"Test : Train\n{len(test_label)} :  {len(train_label)}",end="\n\n")
    
    # Retrieve sample ids to monitor which samples are put in each split 
    if sample_id is not None:
        test_id = sample_id[split_idx_tup[0]]
        train_id = sample_id[split_idx_tup[2]]
        if val_ratio is not None:
            val_id = sample_id[split_idx_tup[1]]
    
    # oversample TRAINING DATA
    original_train_cnt = len(train_label)
    if oversample_type=='ros':
        ros = RandomOverSampler(sampling_strategy='minority', random_state=seed)
        resampled_train_data, resampled_train_label = ros.fit_resample(train_data, train_label)
    elif oversample_type=='smote':
        smote = SMOTE(random_state=seed)
        resampled_train_data, resampled_train_label = smote.fit_resample(train_data, train_label)
    else:
        resampled_train_data, resampled_train_label = train_data, train_label

    # sklearn oversampler matches the number of minor & major label count
    # Only oversample user designated number of samples
    n_original_train_cnt = len(train_label)
    if ratio_or_int(train_minor_label_oversample_ratio)=="ratio":
        n_train_minor_label = len(np.where(train_label==minor_label)[0])
        n_to_oversample = int(n_train_minor_label*train_minor_label_oversample_ratio)
    elif ratio_or_int(train_minor_label_oversample_ratio)=="int":
        n_to_oversample = train_minor_label_oversample_ratio
    else: # "zero" returned
        n_to_oversample = 0
    
    resampled_train_data = resampled_train_data[:n_original_train_cnt + n_to_oversample]
    resampled_train_label = resampled_train_label[:n_original_train_cnt + n_to_oversample]

    if sample_id is not None:
        # Add -1 as train ids for oversampled data points
        train_id = np.concatenate([train_id, [-1]*n_to_oversample])
    
    print("Train size before oversampling: ", len(train_label))
    print("Train size after oversampling: ", len(resampled_train_label),end="\n\n")

    # Again, shuffle resampled training data
    shuffled_idx = np.random.permutation(list(range(len(resampled_train_label))))
    resampled_train_data = resampled_train_data[shuffled_idx]
    resampled_train_label = resampled_train_label[shuffled_idx]
    if sample_id is not None:
        train_id = train_id[shuffled_idx]

    n_test = len(test_label)
    n_train = len(resampled_train_label)
    
    train_minor_label_cnt = np.unique(resampled_train_label, return_counts=True)[1][1] # count of label==1
    train_major_label_cnt = n_train-train_minor_label_cnt
    print(f"Resampled train data's major & minor label ratio:\n{train_major_label_cnt} : {train_minor_label_cnt} = {round(train_major_label_cnt/n_train*100,2)}% : {round(train_minor_label_cnt/n_train*100,2)}%",end="\n\n")
    
    # Convert np arr to pd df if df was inputted
    if data_is_df:
        resampled_train_data = pd.DataFrame(resampled_train_data, columns=df_columns)
        test_data = pd.DataFrame(test_data, columns=df_columns)
        
    if label_is_sr:
        resampled_train_label = pd.Series(resampled_train_label)
        test_label = pd.Series(test_label)
    
    if sample_id_is_sr:
        train_id = pd.Series(train_id)
        test_id = pd.Series(test_id)
        
    # Return split data/label/sample_id
    if val_ratio is None: # without validation set
        new_n_sample = len(test_label)+len(resampled_train_label)
        print(f"Resampled data's test & train ratio:\n{n_test} : {n_train} = {round(n_test/new_n_sample*100,2)}% : {round(n_train/new_n_sample*100,2)}%", end="\n\n")
        if sample_id is None:
            # Without sample id
            return [test_data, test_label,\
                    resampled_train_data, resampled_train_label]
        else:
            # With sample id
            return [test_data, test_label, test_id,
                    resampled_train_data, resampled_train_label, train_id]
    else: # with validation set
        if data_is_df:
            val_data = pd.DataFrame(val_data, columns=df_columns)        
        if label_is_sr:
            val_label = pd.Series(val_label)        

        new_n_sample = len(test_label)+len(val_label)+len(resampled_train_label)
        n_val = len(val_label)
        
        print(f"Resampled data's test & validation & train ratio:\n{n_test} : {n_val} : {n_train} = {round(n_test/new_n_sample*100,2)}% : {round(n_val/new_n_sample*100,2)}% : {round(n_train/new_n_sample*100,2)}%", end="\n\n")
        if sample_id is None:
            # Without sample id
            return [test_data, test_label, \
                    val_data, val_label,\
                    resampled_train_data, resampled_train_label]
        else:
            # With sample id
            if sample_id_is_sr:
                val_id = pd.Series(val_id)
            return [test_data, test_label, test_id,\
                    val_data, val_label, val_id,\
                    resampled_train_data, resampled_train_label, train_id]

### helper functions ###

def make_unique_file_or_dir_name(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f"{filename}({str(counter)}){extension}"
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
    
def fit_logR_model(data, sample_id=None):
    """
    data (2d np array): data & sample id & label
    *second to last righ-most column == sample id
    *last righ-most column == label
    
    returns fitted model
    """
    # columns exteded by 1, which is label
    if sample_id is None: 
        x = data[:, :len(data[0])-1]
        y = data[:,len(data[0])-1]
    # columns exteded by 2, which are sample id & label
    else:
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
        # Use prob of both labels multiplied by gradient to get the "expected" model change (gradient w.r.t. w & b)
        exp_mod_change += model.predict_proba([x])[0][int(m_class)]*(np.linalg.norm((y_hat-m_class)*x) + (y_hat-m_class)**2)
        # model.predict_proba([x])[0][int(m_class)] => indexing to get prob of each class (0 or 1)
    return exp_mod_change

def emc(data, data_pool, sample_id=None):
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
        model = fit_logR_model(np.concatenate([data,[row]],axis=0),sample_id=sample_id)
        
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

def run_l_simulations_emc(data_x, label_y, sample_id=None, **config):
    '''
    data (2d np.array/pd.DataFrame)
    label: 0/1 binary label (1d np.array/pd.Series)
    
    Trains logistic regression (LR), adding 1 sample 
    at a time for each update step using 
    expected model change technique.

    Initially train LR with initial_train_ratio of 
    training data.
    
    *Saves performance metrics for each simulation loop in user designated directory.
    *If sample_id != None, logs samples selected in user designated directory.
    '''
    assert len(data_x) == len(label_y)
    
    n_sim = config['n_sim']
    initial_train_ratio = config['initial_train_ratio']
    log_freq = config['log_freq']
    logger_emc = config['logger_emc']
    emc_dir_path = config['emc_dir_path']
    if config['start_sim'] is None:
        start_sim = 0
    else:
        start_sim = config['start_sim']

    # Simulation loop
    for i in range(start_sim, n_sim):
        logger_emc.info(f'\nEMC simulation round: {i}\n')
        print(f'\nEMC simulation round: {i}\n')

        # Oversample
        test_x, test_y, test_ids, train_x, train_y, train_ids = \
            test_val_train_split_w_oversampling(data=data_x, 
                                                label=label_y, 
                                                sample_id=sample_id,
                                                oversample_type=config['oversample_type'],
                                                test_ratio=config['test_ratio'],
                                                train_minor_label_oversample_ratio=config['train_minor_label_oversample_ratio'],
                                                seed=i)
        train_y = train_y.reshape(-1,1)
        test_y = test_y.reshape(-1,1)
        
        if sample_id is not None:
            train_ids_idx = np.arange(len(train_ids)).reshape(-1,1)
            # Save ids as text file
            with open(f'./{emc_dir_path}/train_sample_id.txt', 'w') as f:
                f.write(f"train_id_idx : train_id\n")
                for tr_id_idx, tr_id in zip(train_ids_idx, train_ids):
                    f.write(f"{tr_id_idx} : {tr_id}\n")

            with open(f'./{emc_dir_path}/test_sample_id.txt', 'w') as f:
                for test_id in test_ids:
                    f.write(f"{test_id}\n")
                    
        logger_emc.info(f'train data shape: {train_x.shape}')
        logger_emc.info(f'test data shape: {test_x.shape}\n')

        # initial random sampling
        init_size = int(len(train_x) * initial_train_ratio)
        assert init_size >= 10,\
        "Initial training size less than 10. Increase initial_train_ratio param \
        or increase number of samples." 
        
        n_updates = len(train_x)-init_size

        # attach sample id & label
        if sample_id is not None:
            data_w_label = np.concatenate([train_x, train_ids_idx, train_y], axis=1)
        else:
            data_w_label = np.concatenate([train_x, train_y], axis=1)

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
        first_model=fit_logR_model(data,sample_id=sample_id)
        accuracy, precision, recall = calc_acc_prec_rec(first_model, test_x, test_y)
        ith_sim_accuracy_updates_ls.append(accuracy)
        ith_sim_precision_updates_ls.append(precision)
        ith_sim_recall_updates_ls.append(recall)
        logger_emc.info(f"\n initial accuracy: {accuracy}\n initial precision: {precision}\n initial recall: {recall}\n")
        ##############################################################
        
        # update n times, choosing one sample at each loop from data_pool
        for j in range(n_updates):
            # sample one using expected model change query.
            sampled_x, data_pool = emc(data, data_pool, sample_id=sample_id)
            if sample_id is not None:
                # log id of the selected sample (last column: label, second to last column: sample id's idx)
                logger_emc.info(f"Selected sample id: {train_ids[int(sampled_x[len(sampled_x)-2])]}")
                ith_sim_id_update_order_ls.append(train_ids[int(sampled_x[len(sampled_x)-2])])
            
            # add selected sample to data
            data = np.concatenate([data, [sampled_x]], axis=0)
            
            # fit new model and measure performance on test set
            updated_model=fit_logR_model(data,sample_id=sample_id)
            accuracy, precision, recall = calc_acc_prec_rec(updated_model, test_x, test_y)

            #append curr update round, and curr accuracy as tuple
            ith_sim_accuracy_updates_ls.append(accuracy)
            ith_sim_precision_updates_ls.append(precision)
            ith_sim_recall_updates_ls.append(recall)
            if j!=0 and (j+1)%log_freq==0:
                logger_emc.info(f'''{j+1}th update\n\
                 data shape: {data.shape}\n\
                 data_pool shape: {data_pool.shape}\n\
                 accuracy: {accuracy}\n\
                 precision: {precision}\n\
                 recall: {recall}\n''')

        # Save metrics inside simulation loop in case loop gets interrupted 
        np.save(f'./{emc_dir_path}/{i}_sim_emc_accuracy.npy', ith_sim_accuracy_updates_ls)
        np.save(f'./{emc_dir_path}/{i}_sim_emc_precision.npy', ith_sim_precision_updates_ls)
        np.save(f'./{emc_dir_path}/{i}_sim_emc_recall.npy', ith_sim_recall_updates_ls)
        if sample_id is not None:
            np.save(f'./{emc_dir_path}/{i}_sim_emc_sample_id.npy', ith_sim_id_update_order_ls)

### Random sampling functions ###

def random_selection_of_sample(data_pool, seed):
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
    data (2d np.array/pd.DataFrame)
    label: 0/1 binary label (1d np.array/pd.Series)
    
    Trains logistic regression (LR), adding 1 sample 
    randomly at a time for each update step.
    
    Initially train LR with initial_train_ratio of 
    training data.
    
    *Saves performance metrics for each simulation loop in user designated directory.
    '''
    assert len(data_x) == len(label_y)
    
    n_sim = config['n_sim']
    initial_train_ratio = config['initial_train_ratio']
    log_freq = config['log_freq']
    logger_rand_sampler = config['logger_rand_sampler']
    rand_sampling_dir_path = config['rand_sampling_dir_path']
    if config['start_sim'] is None:
        start_sim = 0
    else:
        start_sim = config['start_sim']
    # Simulation loop
    for i in range(start_sim,n_sim):
        logger_rand_sampler.info(f'\nRandom sampling simulation round: {i}\n')
        print(f'Random sampling simulation round: {i}\n')

        # Oversample
        test_x, test_y, train_x, train_y= \
            test_val_train_split_w_oversampling(data=data_x,
                                                label=label_y, 
                                                oversample_type=config['oversample_type'],
                                                test_ratio=config['test_ratio'],
                                                train_minor_label_oversample_ratio=config['train_minor_label_oversample_ratio'], 
                                                seed=i)

        train_y = train_y.reshape(-1,1)
        test_y = test_y.reshape(-1,1)
        logger_rand_sampler.info(f'train data shape: {train_x.shape}')
        logger_rand_sampler.info(f'test data shape: {test_x.shape}\n')

        # initial random sampling
        init_size = int(len(train_x) * initial_train_ratio)

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
            sampled_x, data_pool = random_selection_of_sample(data_pool, seed=i)
            # add selected sample to data
            data = np.concatenate([data, [sampled_x]], axis=0)
            
            # fit new model and measure performance on test set
            updated_model=fit_logR_model(data)
            accuracy, precision, recall = calc_acc_prec_rec(updated_model, test_x, test_y)

            #append curr update round, and curr accuracy as tuple
            ith_sim_accuracy_updates_ls.append(accuracy)
            ith_sim_precision_updates_ls.append(precision)
            ith_sim_recall_updates_ls.append(recall)
            
            if j!=0 and (j+1)%log_freq==0:
                logger_rand_sampler.info(f'''{j+1}th update\n\
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
        