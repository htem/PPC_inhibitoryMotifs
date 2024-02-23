import pickle
import pandas as pd
import numpy as np
import h5py 
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sys
sys.path.append('/Users/akuan/repos/ppc_project_analysis/tracing_analysis')
import ppc_analysis_functions.figure_plotting as figs

workingDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/'
sessionsDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/fromDan/'

def gen_LD187_trialData(workingDir):
    with open(workingDir + 'MN_DF_all_PPC.pkl' , 'rb') as f:  
        MN_DF_all = pickle.load(f)

    mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213','LD187_141212','LD187_141211','LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206']
    #mySessions = ['LD187_141216','LD187_141215']
    my_DF_out = MN_DF_all[['matched_cell_ID','skeleton_id','type','select_idx_MI','Ca_trial_mean_bR','Ca_trial_mean_wL','sessions_ROI','choiceMI_max_idx']]

    # Load metrics from neuron dataframe
    skel_ids = MN_DF_all.skeleton_id.values
    #pref_dir = MN_DF_all.select_idx_MI.values > 0

    # Load sessions data and assemble concatenated trial data
    trialData = {} # initialize dict for output metrics
    for skel_id in skel_ids:
        trialData[skel_id] = {}
        trialData[skel_id]['nAct_norm_t'] = np.empty((63,0))
        trialData[skel_id]['nAct'] = np.array([])
        trialData[skel_id]['cueType'] = np.array([])
        trialData[skel_id]['cueDir'] = np.array([])
        trialData[skel_id]['choice'] = np.array([])
        trialData[skel_id]['isCorrect'] = np.array([])
        trialData[skel_id]['type'] = MN_DF_all[MN_DF_all.skeleton_id==skel_id].type.values[0]

    nAct_norm_all= np.empty((len(skel_ids), len(mySessions)))
    nAct_norm_all[:] = np.nan

    for s_idx,session in enumerate(mySessions): 
        print(session)
        mat_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/trialAlignedData/%s.mat" % session
        with h5py.File(mat_path, 'r') as f: # Load relevant trial data from .mat files
            CaData = np.array(f['trialAlignedData']['CaData'])
            cueType =  np.transpose(np.array(f['trialAlignedData']['cueType']))[0]
            cueDir = np.transpose(np.array(f['trialAlignedData']['cueDir']))[0]
            choice = np.transpose(np.array(f['trialAlignedData']['choice']))[0]
            isCorrect = np.transpose(np.array(f['trialAlignedData']['isCorrect']))[0]

        for n_idx,rois in enumerate(MN_DF_all.sessions_ROI.values): # Loop over each neuron
            roi = rois[s_idx]
            skel_id = skel_ids[n_idx]
            if roi != -1:        
                nAct = CaData[13:,:,roi] # Exclude ITI before
                nAct_mean = np.nanmean(nAct, axis=0) # Avg over timepoints
                nAct_scale = np.nanmean(nAct_mean, axis=0) # Avg over trials for normalization
                nAct_norm = nAct_mean/nAct_scale # Rel act compared to avg act for the session 
                nAct_norm_t = nAct/nAct_scale

                trialData[skel_id]['nAct_norm_t'] = np.concatenate((trialData[skel_id]['nAct_norm_t'], nAct_norm_t), axis=1)
                trialData[skel_id]['nAct'] = np.concatenate((trialData[skel_id]['nAct'], nAct_norm)) # nAct_norm is default
                trialData[skel_id]['cueType'] = np.concatenate((trialData[skel_id]['cueType'], cueType))
                trialData[skel_id]['cueDir'] = np.concatenate((trialData[skel_id]['cueDir'], cueDir))
                trialData[skel_id]['choice'] = np.concatenate((trialData[skel_id]['choice'], choice))
                trialData[skel_id]['isCorrect'] = np.concatenate((trialData[skel_id]['isCorrect'], isCorrect))
    trialData['numCells'] = len(skel_ids)

    return (trialData, skel_ids)

def gen_two_session_trialData(mySessions, keys_path):

    #mySessions = ['DW132_20211021','DW132_20211023']
    numSessions = len(mySessions)
    #mySessions = ['DW132_20211023']
    # Load keys to index matches across sessions

    with h5py.File(keys_path, 'r') as f: # Load relevant trial data from .mat files
            print(f.keys())
            idx_map = np.array(f['optimal_cell_to_index_map'])
    print(idx_map.shape)

    trialData = {} # initialize dict for output metrics
    sessionData = {}
    numCells = idx_map.shape[1]
    mat_path = sessionsDir+"%s.mat" % mySessions[0]
    with h5py.File(mat_path, 'r') as f: # Load relevant trial data from .mat files
        CaData = np.array(f['binnedSpkData'])
        numBins = CaData.shape[1]
    skel_ids = np.arange(numCells)
    for skel_id in skel_ids:
        trialData[skel_id] = {}
        trialData[skel_id]['nAct_norm_t'] = np.empty((numBins,0))
        trialData[skel_id]['nAct'] = np.array([])
        trialData[skel_id]['cueType'] = np.array([])
        trialData[skel_id]['isCorrect'] = np.array([])
        trialData[skel_id]['pref_dir_sess'] = np.array([])
        trialData[skel_id]['numSessFound'] = 0
        trialData[skel_id]['type_sess'] = np.array([])

    # Load sessions data and assemble concatenated trial data
    for s_idx,session in enumerate(mySessions): 
        #print(session)
        mat_path = sessionsDir+"%s.mat" % session
        #print(mat_path)
        sessionData[session] = {}
        with h5py.File(mat_path, 'r') as f: # Load relevant trial data from .mat files
            #print(f.keys())
            test = f
            CaData = np.array(f['binnedSpkData'])
            cueType =  np.array([i[0] for i in np.array(f['trialType'])])
            isCorrect = np.array([i[0] for i in np.array(f['isCorrect'])])
            isInh = np.array(f['isRed'][0])
            isExc = np.array(f['notRed'][0])
            numCells = CaData.shape[0]
            numBins = CaData.shape[1]
            numTrials = CaData.shape[2]

            for skel_id in skel_ids:
                sessionData[session][skel_id] = {}
                roi = int(idx_map[s_idx,skel_id] - 1) # correct for MATLAB indexing!!
                if roi !=-1: # index (matlab) 0 is not found          
                    nAct = CaData[roi,:,:] # 
                    nAct_mean = np.nanmean(nAct, axis=0) # Avg over timepoints
                    nAct_scale = np.nanmean(nAct_mean, axis=0) # Avg over trials for normalization
                    nAct_norm = nAct_mean/nAct_scale # Rel act compared to avg act for the session 
                    nAct_norm_t = nAct/nAct_scale # For plotting, avg over trials but not timepoints

                    l_trials = cueType == 1
                    r_trials = cueType == 2
                    if np.mean(nAct_mean[l_trials]) > np.mean(nAct_mean[r_trials]):
                        pref_dir = [1]
                    else:
                        pref_dir = [2]
                    
                    sessionData[session][skel_id]['nAct_norm_t'] = nAct_norm_t

                    trialData[skel_id]['numSessFound'] = trialData[skel_id]['numSessFound']+1 # increment sess found counter
                    trialData[skel_id]['nAct_norm_t'] =  np.concatenate((trialData[skel_id]['nAct_norm_t'], nAct_norm_t), axis=1)
                    trialData[skel_id]['nAct'] =  np.concatenate((trialData[skel_id]['nAct'], nAct_norm)) # nAct_norm is default
                    trialData[skel_id]['cueType'] = np.concatenate((trialData[skel_id]['cueType'], cueType))
                    trialData[skel_id]['isCorrect'] =  np.concatenate((trialData[skel_id]['isCorrect'], isCorrect))
                    trialData[skel_id]['pref_dir_sess'] = np.concatenate((trialData[skel_id]['pref_dir_sess'], pref_dir))

                    if isInh[roi]:
                        trialData[skel_id]['type_sess'] = np.concatenate((trialData[skel_id]['type_sess'], ['non pyramidal']))
                    elif isExc[roi]:
                        trialData[skel_id]['type_sess'] = np.concatenate((trialData[skel_id]['type_sess'], ['pyramidal']))
                    else:
                        trialData[skel_id]['type_sess'] = np.concatenate((trialData[skel_id]['type_sess'], ['unknown']))
            print('Loading mouse %s session %s' % (mouse, session))
    print('%i error trials out of %i' % (np.sum(isCorrect == 0), len(isCorrect)))

    # Load sessions trial data
    trialData['numCells'] = numCells

    # Calculate session consensus 

    # Number of cells found in both sessions
    both = np.array([trialData[i]['numSessFound'] for i in skel_ids]) == 2 
    print('%i of %i cells in both sessions ' % (sum(both), len(both)))

    # Cell types found in both sessions
    sessTypes = np.array([trialData[i]['type_sess'] for i in skel_ids[both]])
    sessTypesAgree = np.array([sessTypes[i][0] == sessTypes[i][1] for i in np.arange(len(sessTypes))])
    print('%i of %i have matching types ' % (sum(sessTypesAgree), len(sessTypesAgree)))

    # Calc cell prefs (but use all cells, calc pref from concat trials)
    cell_prefs = np.array([trialData[i]['pref_dir_sess'] for i in skel_ids[both]])
    match_pref = [cell_prefs[i][0] == cell_prefs[i][1] for i in np.arange(len(cell_prefs))] 
    print('%i of %i have matching pref ' % (sum(match_pref), len(match_pref)))

    # Calc cell consensus fields
    for skel_id in skel_ids:
        # Pref dir
        l_trials = trialData[skel_id]['cueType'] == 1 
        r_trials = trialData[skel_id]['cueType'] == 2
        if np.mean(trialData[skel_id]['nAct'][l_trials]) > np.mean(trialData[skel_id]['nAct'][r_trials]):
            trialData[skel_id]['pref_dir'] = 1
        else:
            trialData[skel_id]['pref_dir'] = 2
        # Cell type if there are mismatches
        if trialData[skel_id]['numSessFound'] > 1 and trialData[skel_id]['type_sess'][0] != trialData[skel_id]['type_sess'][1]:
            if 'non pyramidal' in trialData[skel_id]['type_sess']: # priority goes to inhibitory label
                trialData[skel_id]['type'] = 'non pyramidal' 
            elif 'pyramidal' in trialData[skel_id]['type_sess']:
                trialData[skel_id]['type'] = 'pyramidal'
            else:
                trialData[skel_id]['type'] = 'unknown'   
        #elif not trialData[skel_id]['type_sess']:
        #    trialData[skel_id]['type'] = 'unknown'
        else:
            trialData[skel_id]['type'] = trialData[skel_id]['type_sess'][0]

    match0 = [trialData[i]['pref_dir'] == trialData[i]['pref_dir_sess'][0] for i in skel_ids[both]]
    print('%i of %i have matching pref btw one sess0 and consensus ' % (sum(match0), len(match0)))
    match1 = [trialData[i]['pref_dir'] == trialData[i]['pref_dir_sess'][1] for i in skel_ids[both]]
    print('%i of %i have matching pref btw one sess1 and consensus ' % (sum(match1), len(match1)))

    corrects = trialData[1]['isCorrect']
    print('%i error trials out of %i' % (np.sum(corrects == 0), len(corrects)))
    return trialData

def gen_single_session_trialData(mat_path):
    with h5py.File(mat_path, 'r') as f: # Load relevant trial data from .mat files
        CaData = np.array(f['binnedSpkData'])
        cueType =  np.array([i[0] for i in np.array(f['trialType'])])
        isCorrect = np.array([i[0] for i in np.array(f['isCorrect'])])
        isInh = np.array(f['isRed'][0])
        isExc = np.array(f['notRed'][0])
        numCells = CaData.shape[0]
        numBins = CaData.shape[1]
        numTrials = CaData.shape[2]
    print('Loading mouse %s session %s' % (mouse, session))
    print('%i error trials out of %i' % (np.sum(isCorrect == 0), len(isCorrect)))

    # Load sessions trial data
    trialData = {} # initialize dict for output metrics
    trialData['numCells'] = numCells
    trialData['numBins'] = numBins
    trialData['numTrials'] = numTrials

    skel_ids = np.arange(numCells)
    for skel_id in skel_ids:
        trialData[skel_id] = {}
        trialData[skel_id]['nAct_norm_t'] = np.empty((numBins,0))
        trialData[skel_id]['nAct'] = np.array([])
        trialData[skel_id]['cueType'] = np.array([])
        trialData[skel_id]['isCorrect'] = np.array([])
        #trialData[skel_id]['pref_dir'] = np.array([])
    
        nAct_all = CaData[skel_id,:,:] # 
        nAct_mean = np.nanmean(nAct_all, axis=0) # Avg over timepoints
        nAct_scale = np.nanmean(nAct_mean, axis=0) # Avg over trials for normalization
        if nAct_scale > 0:
            nAct_norm = nAct_mean/nAct_scale # Rel act compared to avg act for the session 
            nAct_norm_t = nAct_all/nAct_scale
        else:
            nAct_norm = np.zeros(nAct_mean.shape)
            nAct_norm_t = np.zeros(nAct_all.shape)

        #l_trials = cueType == 1
        #r_trials = cueType == 2
        #if np.mean(nAct_mean[l_trials]) > np.mean(nAct_mean[r_trials]):
        #    pref_dir = 1
        #else:
        #    pref_dir = 2

        trialData[skel_id]['nAct_norm_t'] = nAct_norm_t
        trialData[skel_id]['nAct'] =  nAct_norm
        trialData[skel_id]['cueType'] = cueType
        trialData[skel_id]['isCorrect'] = isCorrect
        #trialData[skel_id]['pref_dir'] = pref_dir

        # Count how many trials have activitiy
        trialData[skel_id]['perc_active'] = sum(nAct_mean > 0)/len(nAct_mean)

        # Load cell types from label
        if isInh[skel_id]:
            trialData[skel_id]['type'] = 'non pyramidal'
        elif isExc[skel_id]:
            trialData[skel_id]['type'] = 'pyramidal'
        else:
            trialData[skel_id]['type'] = 'unknown'
    return trialData

def gen_errDiff_metrics(trialData, n_shuf = 100, skel_ids = None, cues = 'DW'):
    if skel_ids is None:
        skel_ids = np.arange(trialData['numCells'])
    for skel_id in skel_ids: 
        nAct_norm_t = trialData[skel_id]['nAct_norm_t']
        nAct = trialData[skel_id]['nAct']
        #pref = trialData[skel_id]['pref_dir']
        cueType = trialData[skel_id]['cueType']
        #cueDir = cueType
        #cueDir = trialData[skel_id]['cueDir']
        isCorrect = trialData[skel_id]['isCorrect']
        isError = np.logical_not(isCorrect)
        
        if cues == 'DW':
            cueDir = cueType
            orig =  np.logical_or(cueType == 1, cueType == 2) # orig 2 cues
            og_correct =  np.logical_and(orig, isCorrect) # orig 2 cues
            l_trials = cueType == 1
            r_trials = cueType == 2
            l_correct = np.logical_and(cueType == 1, isCorrect)
            r_correct = np.logical_and(cueType == 2, isCorrect)
            if np.mean(nAct[l_correct]) > np.mean(nAct[r_correct]):
                pref = 1
            else:
                pref = 2

        elif cues == 'LD':
            cueDir = trialData[skel_id]['cueDir']
            orig =  np.logical_or(cueType == 2, cueType == 3) # orig 2 cues
            og_correct =  np.logical_and(orig, isCorrect) # orig 2 cues
            l_trials = cueType == 2
            r_trials = cueType == 3
            l_correct = np.logical_and(cueDir == 0, og_correct)
            r_correct = np.logical_and(cueDir == 1, og_correct)

            if np.mean(nAct[l_correct]) > np.mean(nAct[r_correct]):
                pref = 0
            else:
                pref = 1
        
        #og_correct =  np.logical_and(orig, isCorrect) # orig 2 cues
        og_error =  np.logical_and(orig, isError)
        og_pref =  np.logical_and(orig, cueDir == pref) # pref dir determined by select_idx
        og_pref_correct =  np.logical_and(og_pref, isCorrect) 
        og_pref_error =  np.logical_and(og_pref, isError) 
        og_nonpref =  np.logical_and(orig, np.logical_not(cueDir == pref))
        og_nonpref_correct =  np.logical_and(og_nonpref, isCorrect) 
        og_nonpref_error =  np.logical_and(og_nonpref, isError) 

        # Save average time-courses for plotting 
        trialData[skel_id]['pref_correct_act'] = np.nanmean(nAct_norm_t[:,og_pref_correct], axis=1) #/ np.nanmean(nAct[og_pref])
        trialData[skel_id]['pref_error_act'] = np.nanmean(nAct_norm_t[:,og_pref_error], axis=1) #/ np.nanmean(nAct[og_pref])
        trialData[skel_id]['nonpref_correct_act'] = np.nanmean(nAct_norm_t[:,og_nonpref_correct], axis=1) #/ np.nanmean(nAct[og_nonpref])
        trialData[skel_id]['nonpref_error_act'] = np.nanmean(nAct_norm_t[:,og_nonpref_error], axis=1) #/ np.nanmean(nAct[og_nonpref])

        # Count number of each type of trial (for reference)
        trialData[skel_id]['n_trials'] = sum(orig)
        trialData[skel_id]['n_trials_pref_correct'] = sum(og_pref_correct)
        trialData[skel_id]['n_trials_pref_error'] = sum(og_pref_error)
        trialData[skel_id]['n_trials_nonpref_correct'] = sum(og_nonpref_correct)
        trialData[skel_id]['n_trials_nonpref_error'] = sum(og_nonpref_error)

        if sum(og_pref_error) > 0 and sum(og_nonpref_error) > 0: # only include neurons with >0 error trials
            if np.nanmean(nAct[og_pref]) > 0:
                err_diff_pref = (np.nanmean(nAct[og_pref_error])-np.nanmean(nAct[og_pref_correct]))/np.nanmean(nAct[og_pref])
            else:
                err_diff_pref = 0
            if np.nanmean(nAct[og_nonpref]) > 0:
                err_diff_nonpref = (np.nanmean(nAct[og_nonpref_error])-np.nanmean(nAct[og_nonpref_correct]))/np.nanmean(nAct[og_nonpref])
            else:
                err_diff_nonpref = 0
            if np.nanmean(nAct[og_error]) > 0:
                pref_diff_err = (np.nanmean(nAct[og_nonpref_error])-np.nanmean(nAct[og_pref_error]))/np.nanmean(nAct[og_error])
            else:
                pref_diff_err = 0
            if np.nanmean(nAct[og_correct]) > 0:
                pref_diff_corr = (np.nanmean(nAct[og_pref_correct])-np.nanmean(nAct[og_nonpref_correct]))/np.nanmean(nAct[og_correct])
            else: 
                pref_diff_corr = 0
            err_diff_combined = err_diff_nonpref - err_diff_pref
            
            # Permutation test: shuffle identity of error trials
            err_diff_pref_shuf = np.empty((n_shuf))
            err_diff_nonpref_shuf = np.empty((n_shuf))
            err_diff_combined_shuf = np.empty((n_shuf))
            for ix in range(n_shuf):
                isCorrect_shuf =  np.random.permutation(isCorrect) # Shuffle error trials
                #isCorrect_shuf = isCorrect
                isError_shuf = np.logical_not(isCorrect_shuf)
                if cues == 'DW':
                    orig =  np.logical_or(cueType == 1, cueType == 2) # orig 2 cues
                elif cues == 'LD':
                    orig =  np.logical_or(cueType == 2, cueType == 3) # orig 2 cues
                og_correct =  np.logical_and(orig, isCorrect_shuf) # orig 2 cues
                og_error =  np.logical_and(orig, isError_shuf)
                og_pref =  np.logical_and(orig, cueDir == pref) # pref dir determined by select_idx
                og_pref_correct =  np.logical_and(og_pref, isCorrect_shuf) 
                og_pref_error =  np.logical_and(og_pref, isError_shuf) 
                og_nonpref =  np.logical_and(orig, np.logical_not(cueDir == pref))
                og_nonpref_correct =  np.logical_and(og_nonpref, isCorrect_shuf) 
                og_nonpref_error =  np.logical_and(og_nonpref, isError_shuf) 

                #nAct_shuf = np.random.permutation(nAct) # Shuffle activity across trials
                if np.nanmean(nAct[og_pref]) > 0:
                    err_diff_pref_shuf[ix] = (np.nanmean(nAct[og_pref_error])-np.nanmean(nAct[og_pref_correct]))/np.nanmean(nAct[og_pref])
                else:
                    err_diff_pref_shuf[ix] = 0
                if np.nanmean(nAct[og_nonpref]) > 0:
                    err_diff_nonpref_shuf[ix] = (np.nanmean(nAct[og_nonpref_error])-np.nanmean(nAct[og_nonpref_correct]))/np.nanmean(nAct[og_nonpref])
                else:
                    err_diff_nonpref_shuf[ix] = 0
                err_diff_combined_shuf[ix] = err_diff_nonpref_shuf[ix] - err_diff_pref_shuf[ix]
            
            trialData[skel_id]['err_diff_pref'] = err_diff_pref
            trialData[skel_id]['err_diff_nonpref'] = err_diff_nonpref
            trialData[skel_id]['err_diff_pref_shuf'] = err_diff_pref_shuf
            trialData[skel_id]['err_diff_nonpref_shuf'] = err_diff_nonpref_shuf
            trialData[skel_id]['err_diff_pref_prc'] = stats.percentileofscore(err_diff_pref_shuf, err_diff_pref)
            trialData[skel_id]['err_diff_nonpref_prc'] = stats.percentileofscore(err_diff_nonpref_shuf, err_diff_nonpref)

            trialData[skel_id]['pref_diff_err'] = pref_diff_err
            trialData[skel_id]['pref_diff_corr'] = pref_diff_corr

            trialData[skel_id]['err_diff_combined'] = err_diff_combined
            trialData[skel_id]['err_diff_combined_shuf'] = err_diff_combined_shuf
            trialData[skel_id]['err_diff_combined_prc'] = stats.percentileofscore(err_diff_combined_shuf, err_diff_combined)
        else:
            trialData[skel_id].update({'err_diff_pref':np.nan,'err_diff_nonpref':np.nan,'err_diff_pref_shuf':np.nan,
            'err_diff_nonpref_shuf':np.nan,'err_diff_pref_prc':np.nan, 'err_diff_nonpref_prc':np.nan, 'pref_diff_err':np.nan,
            'pref_diff_corr':np.nan, 'err_diff_combined':np.nan, 'err_diff_combined_shuf':np.nan,'err_diff_combined_prc':np.nan})

    # Save to datafarme
    MN_DF_all = pd.DataFrame()
    MN_DF_all['neuron'] = skel_ids
    MN_DF_all['type'] = [trialData[i]['type'] for i in skel_ids] 
    MN_DF_all['n_shuf'] = n_shuf


    MN_DF_all['n_trials'] = [trialData[i]['n_trials'] for i in skel_ids]
    MN_DF_all['n_trials_pref_correct'] = [trialData[i]['n_trials_pref_correct'] for i in skel_ids] 
    MN_DF_all['n_trials_pref_error'] = [trialData[i]['n_trials_pref_error'] for i in skel_ids] 
    MN_DF_all['n_trials_nonpref_correct'] = [trialData[i]['n_trials_nonpref_correct'] for i in skel_ids] 
    MN_DF_all['n_trials_nonpref_error'] = [trialData[i]['n_trials_nonpref_error'] for i in skel_ids] 

    MN_DF_all['pref_correct_act'] = [trialData[i]['pref_correct_act'] for i in skel_ids] 
    MN_DF_all['pref_error_act'] = [trialData[i]['pref_error_act'] for i in skel_ids] 
    MN_DF_all['nonpref_correct_act'] = [trialData[i]['nonpref_correct_act'] for i in skel_ids] 
    MN_DF_all['nonpref_error_act'] = [trialData[i]['nonpref_error_act'] for i in skel_ids] 

    MN_DF_all['err_diff_pref'] = [trialData[i]['err_diff_pref'] for i in skel_ids]
    MN_DF_all['err_diff_nonpref'] = [trialData[i]['err_diff_nonpref'] for i in skel_ids]
    MN_DF_all['err_diff_pref_shuf'] = [trialData[i]['err_diff_pref_shuf'] for i in skel_ids]
    MN_DF_all['err_diff_nonpref_shuf'] = [trialData[i]['err_diff_nonpref_shuf'] for i in skel_ids]
    MN_DF_all['err_diff_pref_prc'] = [trialData[i]['err_diff_pref_prc'] for i in skel_ids]
    MN_DF_all['err_diff_nonpref_prc'] = [trialData[i]['err_diff_nonpref_prc'] for i in skel_ids]

    MN_DF_all['pref_diff_err'] = [trialData[i]['pref_diff_err'] for i in skel_ids]
    MN_DF_all['pref_diff_corr'] = [trialData[i]['pref_diff_corr'] for i in skel_ids]

    MN_DF_all['err_diff_combined'] = [trialData[i]['err_diff_combined'] for i in skel_ids]
    MN_DF_all['err_diff_combined_shuf'] = [trialData[i]['err_diff_combined_shuf'] for i in skel_ids]
    MN_DF_all['err_diff_combined_prc'] = [trialData[i]['err_diff_combined_prc'] for i in skel_ids]

    #MN_DF_all['perc_active'] = [trialData[i]['perc_active'] for i in skel_ids] 
    return MN_DF_all

# Calc and save dataframes
n_shuf = 1000

#LD187 11 sessions
mouse = 'LD187'
session = '11'
(trialData, skel_ids) = gen_LD187_trialData(workingDir)
LD187_DF = gen_errDiff_metrics(trialData, skel_ids=skel_ids, cues = 'LD', n_shuf=n_shuf)
out_path = workingDir + mouse+'_'+session+'.pkl'
with open(out_path, 'wb') as f:  
    pickle.dump(LD187_DF, f)
    
#DW177_20220923
mouse = 'DW177'
session = '20220923'
my_path = sessionsDir+mouse+'/'+mouse + '_' + session + '.mat'
my_MN_DF_all = gen_errDiff_metrics(gen_single_session_trialData(my_path), n_shuf = n_shuf)
out_path = workingDir + mouse+'_'+session+'.pkl'
with open(out_path, 'wb') as f:  
    pickle.dump(my_MN_DF_all, f)

#DW154_20220426
mouse = 'DW154'
session = '20220426'
my_path = sessionsDir+mouse+'/'+mouse + '_' + session + '.mat'
my_MN_DF_all = gen_errDiff_metrics(gen_single_session_trialData(my_path), n_shuf = n_shuf)
out_path = workingDir + mouse+'_'+session+'.pkl'
with open(out_path, 'wb') as f:  
    pickle.dump(my_MN_DF_all, f)

#DW132_20211021 DW132_20211023
mouse = 'DW132'
session = '20211021_20211023'
mySessions = ['DW132/DW132_20211021','DW132/DW132_20211023']
keys_path = sessionsDir+mouse+'/'+"optimalMatchIndices.mat"
# Different code for 2 aligned sessions
my_MN_DF_all = gen_errDiff_metrics(gen_two_session_trialData(mySessions, keys_path), n_shuf = n_shuf)
#my_MN_DF_all['session'] = mouse+'_'+session
out_path = workingDir + mouse+'_'+session+'.pkl'
with open(out_path, 'wb') as f:  
    pickle.dump(my_MN_DF_all, f)
