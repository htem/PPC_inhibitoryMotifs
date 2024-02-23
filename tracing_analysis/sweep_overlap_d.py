# Sweep overlap threshold
import sys
import navis
import pymaid
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats as stats

import pandas as pd
import sys
from datetime import date
import pickle
import time 
import logging, sys
from scipy.spatial import distance as dist
import networkx as nx
from sklearn.linear_model import LinearRegression

import importlib
import ppc_analysis_functions.catmaid_API as cAPI
importlib.reload(cAPI)
import ppc_analysis_functions.figure_plotting as figs
importlib.reload(figs)
import analysis_dataframes as myDF
importlib.reload(myDF)
import random
import h5py 
import itertools

# connect to pymaid instance
rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff')
labels = pymaid.get_label_list()
mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213']
workingDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/'
figsDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/figures_working/rev1/'

max_dists = np.logspace(0,2,num=25)

#max_dists = np.logspace(-1,2,num=5)
#max_dists = [2,3.5,5,10,20]

n_shuf = 1000


def calc_bootstrap_data_corr(df,x='pair_select_idx', y='syn_den', sig_test='pearson'):
    boot_df = df.sample(frac=1, replace=True, axis='rows')
    (c,p) = figs.scatter(boot_df, x=x, y=y, sig_test=sig_test,make_plot=False)
    return c
def calc_shuf_corr_both(df,sig_test='pearson', y='syn_den'):
    df['source_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx.values, index = df.index)
    df['target_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx.values, index = df.index)
    df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx(row['source_select_idx_shuf'], row['target_select_idx_shuf']), axis=1)
    (c,p) = figs.scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
    return c

cn_type = 'E-I' # Toggle
#cn_type = 'I-E'
x = 'pair_select_idx_new'
with open(workingDir+'dir_cn_DF_PPC.pkl', 'rb') as f:  
        dir_cn_DF= pickle.load(f)
dir_cn_DF = dir_cn_DF[dir_cn_DF.cn_type==cn_type]

corr_mat = np.zeros((len(max_dists),5))
for i,max_dist in enumerate(max_dists):
    
    dir_cn_DF.dropna(axis = 0, subset = ['pair_select_idx_new','syn_den'], inplace=True)
    if len(dir_cn_DF) > 1:
        dir_cn_DF = myDF.add_cable_overlaps(dir_cn_DF, max_dist = max_dist*1000)
        (c,p) = figs.scatter(dir_cn_DF, x=x, y='syn_den', color='k',sig_test='pearson',s=50, make_plot=False)
        boot_corrs = [calc_bootstrap_data_corr(dir_cn_DF) for i in range(n_shuf)]
        shuf_corrs = [calc_shuf_corr_both(dir_cn_DF) for i in range(n_shuf)]
        ci_low = np.percentile(boot_corrs, 5) # "one-tailed test"
        ci_high = np.percentile(boot_corrs, 95)
        shuf_low = np.percentile(shuf_corrs, 5)
        shuf_high = np.percentile(shuf_corrs, 95)
        corr_mat[i, :] = [c, ci_low, ci_high, shuf_low, shuf_high]
    else:
        corr_mat[i,:] = [np.nan, np.nan, np.nan, np.nan, np.nan]

sweep_d = {'max_dists':max_dists,'corrs': corr_mat[:,0], 'ci_low': corr_mat[:,1], 'ci_high': corr_mat[:,2],
    'shuf_low':corr_mat[:,3], 'shuf_high':corr_mat[:,4]}

with open(workingDir+'sweepD_pearson_'+cn_type+'_dir_cn_DF_PPC.pkl', 'wb') as f:  
        pickle.dump(sweep_d, f)