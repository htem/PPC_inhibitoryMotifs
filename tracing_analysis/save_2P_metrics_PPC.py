# ATK 210115
# Pull 2P data (as saved from Matlab preprocessing)
# Save relevant metrics as pickle
# ATK 220221 Update to test different trial type delinitions (cue, turn, correct only etc...)
# ATK 220606 add choiceMI_sig_t
import pymaid
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys
import os
import h5py
import pickle
import time
import h5py

mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213','LD187_141212','LD187_141211',
              'LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206']

#mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213']
trialTypes = ['wL_trials','bR_trials']

print('Saving trial-aligned 2P data metrics')
# Load trial-aligned 2P physiology data (new 200512)
Ca_trial_means = dict.fromkeys(mySessions)
RL_selectIdx = dict.fromkeys(mySessions)
SNR_raw = dict.fromkeys(mySessions)
tCOM = dict.fromkeys(mySessions)
trial_snr = dict.fromkeys(mySessions)

corr_raw = dict.fromkeys(mySessions)
corr_RL_concat = dict.fromkeys(mySessions)
corr_trial_avg = dict.fromkeys(mySessions)
corr_residual = dict.fromkeys(mySessions)

#mat_path = '/n/groups/htem/temcagt/datasets/ppc/2P_data/code_workspace/LD187/summaryData/actCorr.mat'
mat_path = "/Volumes/Macintosh HD/Volumes/Aaron's PPC/ppc/2P_data/code_workspace/LD187/summaryData/actCorr.mat"
output_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/select_metrics/"

with h5py.File(mat_path, 'r') as f:
    for session in mySessions:
        t = time.time()    
        
        RL_selectIdx[session]= np.array(f['actCorr'][session]['trialAlignedData']['RL_selectIdx'])
        SNR_raw[session]= np.array(f['actCorr'][session]['trialAlignedData']['SNR_raw'])
        
        corr_raw[session] = np.array(f['actCorr'][session]['trialAlignedData']['corr_all'])
        corr_RL_concat[session] = np.array(f['actCorr'][session]['trialAlignedData']['corr_Ca_concat'])
        corr_residual[session] = np.array(f['actCorr'][session]['trialAlignedData']['corr_Ca_residual'])
        corr_trial_avg[session] = np.array(f['actCorr'][session]['trialAlignedData']['corr_Ca_trialMean'])
        
        Ca_trial_means[session]= dict.fromkeys(trialTypes)
        Ca_trial_means[session]['wL_trials']=np.array(f['actCorr'][session]['trialAlignedData']['wL_trials']['Ca_trialMean'])
        Ca_trial_means[session]['bR_trials']= np.array(f['actCorr'][session]['trialAlignedData']['bR_trials']['Ca_trialMean'])
        
        tCOM[session]= dict.fromkeys(trialTypes)
        tCOM[session]['wL_trials']=np.array(f['actCorr'][session]['trialAlignedData']['wL_trials']['tCOM'])
        tCOM[session]['bR_trials']=np.array(f['actCorr'][session]['trialAlignedData']['wL_trials']['tCOM'])
        
        trial_snr[session]= dict.fromkeys(trialTypes)
        trial_snr[session]['wL_trials']=np.array(f['actCorr'][session]['trialAlignedData']['wL_trials']['trial_snr'])
        trial_snr[session]['bR_trials']=np.array(f['actCorr'][session]['trialAlignedData']['bR_trials']['trial_snr'])
        
        elapsed = time.time() - t
        print('Loading session %s took %s seconds' % (session, elapsed))

    with open(output_path + '2P_data_PPC.pkl', 'wb') as f:  
        pickle.dump([RL_selectIdx, SNR_raw, corr_raw, corr_RL_concat, corr_residual,
                    corr_trial_avg, Ca_trial_means, tCOM, trial_snr], f)

'''
# Import supplemental selectivity measures (selectivity from trial shuffles)
#print('Saving choice selectivity measures (RL, ROC, MI)')
print('Saving choice selectivity measures')

AUC_fullTrial = dict.fromkeys(mySessions)
AUC_blocks = dict.fromkeys(mySessions)
RL_selectivity_ROC = dict.fromkeys(mySessions)
RL_selectivity_ROC_blocks = dict.fromkeys(mySessions)

RL_fullTrial = dict.fromkeys(mySessions)
RL_blocks = dict.fromkeys(mySessions)
RL_selectivity = dict.fromkeys(mySessions)
RL_selectivity_blocks = dict.fromkeys(mySessions)

choiceMI = dict.fromkeys(mySessions)
choiceMI_sig_t = dict.fromkeys(mySessions)
choiceMI_max = dict.fromkeys(mySessions)
choiceMI_max_idx = dict.fromkeys(mySessions)
#maxMI_sig = dict.fromkeys(mySessions)
#choiceMI_prctile = dict.fromkeys(mySessions)
choiceMI_pref = dict.fromkeys(mySessions)

#mat_path = '/Users/akuan/repos/ppc_project_analysis/new_2p_analysis/210630_update/'
mat_path = "/Volumes/Macintosh HD/Volumes/Aaron's PPC/ppc/2P_data/code_workspace/LD187/selectMetrics/"
#mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_220221_correctLR/"
#mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_220221_turnLR/"
#mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_220221_cue/"
#mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_220221_turnLRRnew/"


for session in mySessions:
    path = os.path.join(mat_path, session + '.mat')
    print(path)
    with h5py.File(path, 'r') as f:
        t = time.time()       

        AUC_fullTrial[session]= np.array(f['AUC_fullTrial'])
        AUC_blocks[session]= np.array(f['AUC_blocks'])
        RL_selectivity_ROC[session]= np.array(f['selectivity_AUC'])
        RL_selectivity_ROC_blocks[session]= np.array(f['selectivity_AUC_blocks'])
        
        RL_fullTrial[session]= np.array(f['RL_fullTrial'])
        RL_blocks[session]= np.array(f['RL_blocks'])
        RL_selectivity[session]= np.array(f['selectivity_RL'])
        RL_selectivity_blocks[session]= np.array(f['selectivity_RL_blocks'])

        choiceMI[session] = np.array(f['choiceMI'])
        choiceMI_max[session] = np.array(f['choiceMI_max'])
        choiceMI_max_idx[session] = np.array(f['choiceMI_max_idx'])
        #maxMI_sig[session] = np.array(f['maxMI_sig'])
        #choiceMI_prctile[session] = np.array(f['choiceMI_prctile'])
        choiceMI_pref[session] = np.array(f['choiceMI_pref'])

        elapsed = time.time() - t
        print('Loading session %s took %s seconds' % (session, elapsed))

# Saving the objects
#with open('local_data/2P_data_PPC_220221_correctLR.pkl', 'wb') as f:  
#with open('local_data/2P_data_PPC_220221_turnLR.pkl', 'wb') as f:  
#with open('local_data/2P_data_PPC_220221_cueLR.pkl', 'wb') as f:  
#with open('local_data/2P_data_PPC_220221_turnLRRnew.pkl', 'wb') as f: 

with open(output_path+'2P_data_PPC_210802.pkl', 'wb') as f:  # 230130 no longer used
    #pickle.dump([AUC_fullTrial, AUC_blocks, RL_selectivity_ROC, RL_selectivity_ROC_blocks,
    #    RL_fullTrial, RL_blocks, RL_selectivity, RL_selectivity_blocks,
    #    choiceMI, choiceMI_max, choiceMI_max_idx, maxMI_sig, choiceMI_prctile, choiceMI_pref], f)
    pickle.dump([choiceMI, choiceMI_max, choiceMI_max_idx,  choiceMI_pref], f)
'''

# Load MI selectivity metrics for individual epochs (220625)
# ATK edit for time-synced metrics
# ATK 230606 edit to add sig each timepoint

#mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_220625"
#mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_230130"
#mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_230607"
mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/code_workspace/LD187/selectMetrics_230621"
pair_select_sync = dict.fromkeys(mySessions)
select_idx_t = dict.fromkeys(mySessions)

choiceMI = dict.fromkeys(mySessions)
choiceMI_sig_t = dict.fromkeys(mySessions)
choiceMI_prctile = dict.fromkeys(mySessions)
choiceMI_max = dict.fromkeys(mySessions)
choiceMI_max_idx = dict.fromkeys(mySessions)
choiceMI_pref = dict.fromkeys(mySessions)
maxMI_epochs = dict.fromkeys(mySessions)
choiceMI_pref_epochs = dict.fromkeys(mySessions)
maxMI_epochs_long = dict.fromkeys(mySessions)
choiceMI_pref_epochs_long = dict.fromkeys(mySessions)
for session in mySessions:
    path = os.path.join(mat_path, session + '.mat')
    print(path)
    with h5py.File(path, 'r') as f:
        t = time.time()       

        select_idx_t[session] = np.array(f['select_idx_t'])
        pair_select_sync[session] = dict.fromkeys('pair_select_idx')
        pair_select_sync[session]['pair_select_idx'] = np.array(f['pair_select_sync']['pair_select_idx'])

        choiceMI[session] = np.array(f['choiceMI'])
        choiceMI_sig_t[session] = np.array(f['choiceMI_sig_t'])
        choiceMI_prctile[session] = np.array(f['choiceMI_prctile'])
        choiceMI_max[session] = np.array(f['choiceMI_max'])
        choiceMI_max_idx[session] = np.array(f['choiceMI_max_idx'])
        choiceMI_pref[session] = np.array(f['choiceMI_pref'])
        maxMI_epochs[session] = np.array(f['maxMI_epochs'])
        choiceMI_pref_epochs[session] = np.array(f['choiceMI_pref_epochs'])
        maxMI_epochs_long[session] = np.array(f['maxMI_epochs_long'])
        choiceMI_pref_epochs_long[session] = np.array(f['choiceMI_pref_epochs_long'])
        elapsed = time.time() - t
        print('Loading session %s took %s seconds' % (session, elapsed))

# Saving the objects
#with open(output_path + '2P_data_PPC_220625.pkl', 'wb') as f:  
#with open(output_path + '2P_data_PPC_230607.pkl', 'wb') as f:  
with open(output_path + '2P_data_PPC_230621.pkl', 'wb') as f:  
    pickle.dump([pair_select_sync, select_idx_t, choiceMI, choiceMI_sig_t, choiceMI_prctile, 
    choiceMI_max, choiceMI_max_idx,  choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs,
    maxMI_epochs_long, choiceMI_pref_epochs_long], f)


# Load corr metrics (Pearson and MI) 210804
print('Saving pairwise correlation  metrics (Pearson and mutual information)')
mat_path = "/Volumes/Macintosh HD/Volumes/Aaron's PPC/ppc/2P_data/code_workspace/LD187/corrMetrics"
pearsonCorr = dict.fromkeys(mySessions)
pairMI = dict.fromkeys(mySessions)

for session in mySessions:
    path = os.path.join(mat_path, session + '.mat')
    with h5py.File(path, 'r') as f:
        t = time.time()    
        print(path)
        
        #pairMI[session] = dict.fromkeys(['lr','trialResidual'])
        #pairMI[session]['lr'] = np.array(f['pairMI']['lr'])
        #pairMI[session]['trialResidual'] = np.array(f['pairMI']['trialResidual'])
        
        pearsonCorr[session] = dict.fromkeys(['corr_all','corr_Ca_lr','corr_Ca_trialMean','corr_Ca_residual',
            'corr_trialMean_all','corr_bw_diff','corr_Ca_timeMean'])
        pearsonCorr[session]['corr_all'] = np.array(f['pearsonCorr']['corr_all'])
        pearsonCorr[session]['corr_Ca_lr'] = np.array(f['pearsonCorr']['corr_Ca_lr'])
        pearsonCorr[session]['corr_Ca_trialMean'] = np.array(f['pearsonCorr']['corr_Ca_trialMean'])
        pearsonCorr[session]['corr_Ca_residual'] = np.array(f['pearsonCorr']['corr_Ca_residual'])

        # ATK 220802 add new metrics
        pearsonCorr[session]['corr_trialMean_all'] = np.array(f['pearsonCorr']['corr_trialMean_all'])
        pearsonCorr[session]['corr_trialResidual_bwavg'] = np.array(f['pearsonCorr']['corr_trialResidual_bwavg'])
        pearsonCorr[session]['corr_bw_diff'] = np.array(f['pearsonCorr']['corr_bw_diff'])
        pearsonCorr[session]['corr_timeMean'] = np.array(f['pearsonCorr']['corr_timeMean'])

        # ATK 230127 add new metrics
        pearsonCorr[session]['corr_lr_bin'] = np.array(f['pearsonCorr']['corr_lr_bin'])
        pearsonCorr[session]['frac_coincident'] = np.array(f['pearsonCorr']['frac_coincident'])

        elapsed = time.time() - t
        print('Loading session %s took %s seconds' % (session, elapsed))

    with open(output_path+'2P_data_PPC_corrMetrics_230127.pkl', 'wb') as f:  
        #pickle.dump([pairMI, pearsonCorr], f)
        pickle.dump(pearsonCorr, f)