""" Load UKB data """

from itertools import combinations
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from random import sample
import os
from tqdm import tqdm
import multiprocessing as mtp
import time

def loadtxt_parallelly(files, num_work):
    """files: file names."""
    with mtp.Pool(num_work) as P:
        x = P.map(np.loadtxt, files)
    return np.array(x)

def load_all_data(path):
    ''' Load healthy participants data and their information '''
    infs = np.loadtxt(f'{path}/infs.txt', dtype=np.int32)
    fns = [f'{path}/fcu_regressed/{i}.txt' for i in infs[:,0]] # head motion regressed
    print('Head motion regressed data...')
    t0 = time.time()
    X = loadtxt_parallelly(fns, 14)
    print('Time cost: {:.2f}'.format(time.time()- t0))
    return infs, X

def get_data_inf(infs, age_groups, n_subs, ind_n_subs):
    ''' index (absolute) and information of main and independent data'''
    eid, age, sex = infs.T
    f_m_i, f_m_n = sep_age_sex(age, sex, age_groups) # indexs and numbers
    # ind_n_samples = (f_m_n.min(axis=1) / 10).astype(np.int32)
    main_i, ind_f_m_i, ind_f_m_n = balance_age_sex(f_m_i, f_m_n, n_subs)
    ind_i, _, _ = balance_age_sex(ind_f_m_i, ind_f_m_n, ind_n_subs)
    return main_i, infs[main_i], ind_i, infs[ind_i]

def sep_age_sex(age, sex, age_groups):
    """Separate F & M at each age: e.g., [[fi_48, mi_48], ..., [fi_75, mi_75]]"""
    sep = lambda x, i: np.where(x == i)[0]
    ages = np.hstack([np.arange(*a, dtype=np.int32) for a in age_groups]) # all ages
    fi, mi = [sep(sex, i) for i in [0, 1]] # idx of F & M
    fa, ma = age[fi], age[mi] # age of F & M
    f_m_i, f_m_n = [], []
    for a in ages:
        i0, i1 = [sep(x, a) for x in [fa, ma]]
        [f_m_i.append([fi[i0], mi[i1]]), f_m_n.append([len(i0),len(i1)])] # [f_i, m_i]
    return f_m_i, np.array(f_m_n)

def balance_age_sex(f_m_i, nums, n_subs): # F = M
    """Balance F & M at each age and get the remainder: e.g.\n
    balanced: [fi_48, mi_48,..., fi_75, mi_75]\n
    remainder: [[fi_48, mi_48], ..., [fi_75, mi_75]], numbers"""
    fs = lambda x, n: sample(list(x), n)
    one = np.ones_like(nums, dtype=np.int32)
    if n_subs == None: ns = nums # all data
    elif len(n_subs) == 0: ns = one * nums.min() # same #participants for all age 
    elif len(n_subs) == 1: ns = one * n_subs[0] # same #participants for all age
    else: ns = (one.T * n_subs).T
    #### TODO: nums.min(axis=1), nums.min(axis=0)
    main_i, ind_f_m_i = [], []
    for i01, n01 in zip(f_m_i, ns): # [f_i, m_i], [f_n, m_n]
        fmi = [fs(x, n) for x,n in zip(i01, n01)] # balancing
        _fmi = [sorted(set(x)-set(y)) for x,y in zip(i01, fmi)] # remainder
        [main_i.append(np.hstack(fmi)), ind_f_m_i.append(_fmi)]
    return np.hstack(main_i), ind_f_m_i, nums-ns

def union_age_sex(inf, age_groups):
    """group label represent both age group and sex: e.g.,\n
    0: F(group_1), 1: M(group_1),..., 2n: F(group_n), 2n+1: M(group_n)"""
    _, age, sex = inf.T
    Y_group, sex_mask = np.copy(age), sex > 0.5
    for i,a in enumerate(age_groups):
        j = (age >= a[0]) & (age < a[1]) # an age group
        Y_group[j] = 2*i # F: even
    Y_group[sex_mask] += 1 # M: odd
    return Y_group

def fc_selection(i, feat, n_feat, path):
    if feat == 'all_fc': idx = np.arange(1485, dtype=np.int32)
    else:
        pt = f'{path}/all_fc/important_fc'
        fi, mi, ci = [set(np.loadtxt(f'{pt}/{i}-{n_feat}-{k}.txt', dtype=np.int32))\
            for k in ['f','m','c']]
        if feat == 'female_fc': idx = sorted(fi)
        elif feat == 'male_fc': idx = sorted(mi)
        elif feat == 'common_fc': idx = sorted(ci)
        elif feat == 'common_and_female_fc': idx = sorted(ci | fi)
        else: idx = sorted(ci | mi)
    return idx, len(idx)

def random_split(inf, train_val_test, rs):
    ''' Train, val, test index'''
    a,b,c = train_val_test
    x, y = np.arange(len(inf)), inf[:,1] * 2 + inf[:,2]
    i01, i2, y01, y2 = train_test_split(x, y, test_size=c, stratify=y, random_state=rs)
    i0, i1, y0, y1 = train_test_split(i01, y01, test_size=b/(a+b), stratify=y01, random_state=rs)
    return [i0, i1, i2]

def uk_tasks(age_groups, Y_group, i012):
    ''' All age classification tasks '''
    i012s, tasks, ns = [], [], np.arange(len(age_groups))
    yo = list(combinations(ns, 2))
    age_tasks(age_groups, Y_group, i012, i012s, tasks, yo)
    [tasks.append('all'), i012s.append(i012)] # 14-way (F+M)
    return i012s, tasks

def age_tasks(age_groups, Y_group, i012, i012s, tasks, yo):
    ''' Train, val, test index of each age classification tasks'''
    names, ags = ['f', 'm', 'fm', 'f all', 'm all'], age_groups
    f2 = lambda a,b: set(np.where((Y_group==a)|(Y_group==b))[0]) # idx of 2 groups
    for t in names:
        if t == 'f': idxs = [f2(2*y, 2*o) for y,o in yo] # 2-way (F)
        elif t == 'm': idxs = [f2(2*y+1, 2*o+1) for y,o in yo] # 2-way (M)
        elif t == 'f all': idxs = [set(np.where(Y_group % 2 == 0)[0])] # 7-way (F)
        elif t == 'm all': idxs = [set(np.where(Y_group % 2 == 1)[0])] # 7-way (M)
        else: idxs = [f2(2*y, 2*o) | f2(2*y+1, 2*o+1) for y,o in yo] # 2-way (FM)
        tasks += [f'{t}'] if len(idxs)==1 else [f'{t}-{ags[y]}vs{ags[o]}' for y,o in yo]
        i012s += [[np.array(list(idx&set(i))) for i in i012] for idx in idxs]
    [tasks.append('all'), i012s.append(i012)] # 7-way (FM)

def get_fold_data(out_path, i012s, ind_i012s, tasks, inner_i, Y_group, ind_Y_group, main_x, ind_x, batch_size, main_inf, ind_inf):
    ''' Get train, val, test, ind data of each fold '''
    for i012, i3, task in zip(i012s, ind_i012s, tasks):
        if task[:2] in ['f-', 'm-']:
            result_path = f'{out_path}/{task}'
            train_val_test_inf = [main_inf[i] for i in i012]
            save_train_val_test_inf(result_path, inner_i, train_val_test_inf + [ind_inf[i3[0]]])
            y = Y_group2label(task, Y_group, i012[0])
            ind_y = Y_group2label(task, ind_Y_group, i3[0])
            X, y = torch.FloatTensor(np.copy(main_x)), torch.LongTensor(y)
            ind_X, ind_y = torch.FloatTensor(np.copy(ind_x)), torch.LongTensor(ind_y)
            loaders = data_loader(X, y, i012, batch_size)
            ind_loaders = data_loader(ind_X, ind_y, i3, batch_size)
            yield loaders + ind_loaders, task, result_path

def Y_group2label(task, Y_group, train_idx):
    ''' Group label to classification label'''
    y = np.copy(Y_group)
    if task in ['all', 'f all', 'm all']: y = y//2 # 7-way
    elif task.split('-')[0] in ['fm', 'f', 'm']: # 2-way
        y = y//2 # for FM
        for v,k in enumerate(np.unique(y[train_idx])): y[y==k] = v
    else: y = y # all
    return y

def data_loader(X, Y, i012, batch_size):
    """train, valid, and test data loader"""
    loaders, shuffle = [], True
    for idx in i012:
        d = [(x,y) for x,y in zip(np.copy(X[idx]), np.copy(Y[idx]))]
        loader = DataLoader(d, batch_size=batch_size, shuffle=shuffle)
        loaders.append(loader)
        shuffle = False # only shuffle the training set
    return loaders

def save_main_ind_inf(path, main_inf, ind_inf):
    os.makedirs(path, exist_ok=True)
    keys, infs = ['main','ind'], [main_inf, ind_inf]
    [np.savetxt(f'{path}/{k}.txt', v, fmt='%d') for k,v in zip(keys, infs)]

def save_train_val_test_inf(path, inner_i, train_val_test_inf):
    os.makedirs(path, exist_ok=True)
    [np.savetxt(f'{path}/{n}_{inner_i}.txt', v, fmt='%d') for n,v in\
        zip(['train','val','test','ind'], train_val_test_inf)]

#### cross validation
def load_ukb(args):
    infs, X = load_all_data(args.data_path)
    # repeat n times
    for oi in range(0, args.out_n):
        out_path = f'{args.result_path}/{args.feat}/{oi}'
        random.seed(oi), np.random.seed(oi)
        main_i, main_inf, ind_i, ind_inf = get_data_inf(infs, args.age_groups, args.n_subs, args.ind_n_subs)
        save_main_ind_inf(out_path, main_inf, ind_inf)
        Y_group, ind_Y_group = [union_age_sex(inf, args.age_groups) for inf in [main_inf, ind_inf]]
        fc_idx, in_dim = fc_selection(oi, args.feat, args.top_n, args.result_path)
        main_x, ind_x = np.copy(X[main_i][:,fc_idx]), np.copy(X[ind_i][:,fc_idx])
        for ii in range(args.in_n):
            i012 = random_split(main_inf, args.train_val_test_ratio, ii)
            i012s, tasks = uk_tasks(args.age_groups, Y_group, i012)
            ind_i012s, _ = uk_tasks(args.age_groups, ind_Y_group, [np.arange(len(ind_x))])
            loaders_task = get_fold_data(out_path, i012s, ind_i012s, tasks, ii, Y_group, ind_Y_group, main_x, ind_x, args.batch_size, main_inf, ind_inf)
            for _ in range(21*2): # 21 tasks * 2
                loaders, task_name, result_path = next(loaders_task)
                yield loaders, task_name, ii, result_path, in_dim
