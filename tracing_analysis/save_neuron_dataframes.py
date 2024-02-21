# Save neuron dataframes master
import ppc_analysis_functions.catmaid_API as cAPI
import analysis_dataframes as myDF
import pickle 
import numpy as np
import h5py

allSessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213','LD187_141212','LD187_141211',
        'LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206']


def save_conv_DF(dataset, output_file, neurons_DF, cn_DF, psp_DF, n_shuf=0):
    conv_pair_DF= myDF.gen_conv_pair_DF(neurons_DF, cn_DF, psp_DF, n_shuf=n_shuf, 
        dataset=dataset, select_idx=select_idx, selectivity=selectivity)
    with open(output_file, 'wb') as f:
        pickle.dump(conv_pair_DF, f)
 
def load_PPC_actData(mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213']):
    #blocks = ['cueEarly','cueLate','delayEarly','delayTurn','turnITI']
    blocks = None
    #mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213']
    actData = {}
    actData['mySessions'] = mySessions

    # Load 2P data metrics # TODO reorganize 2p data as dict
    with open(load_2P_path+'2P_data_PPC.pkl', 'rb') as f:  
        RL_selectIdx, SNR_raw, corr_raw, corr_RL_concat, corr_residual, corr_trial_avg, Ca_trial_means, tCOM, trial_snr = pickle.load(f)
    (actData['corr_raw'],actData['corr_trial_avg'],actData['corr_residual'],
        actData['trial_snr']) = (corr_raw, corr_trial_avg, corr_residual, trial_snr)
    actData['Ca_trial_means'] = Ca_trial_means
    (actData['tCOM']) = (tCOM)
    
    # choice selectivity metrics (RL, ROC, MI)
    '''
    with open(load_2P_path+'2P_data_PPC_220625.pkl', 'rb') as f:  
        choiceMI, choiceMI_max, choiceMI_max_idx,  choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs, maxMI_epochs_long, choiceMI_pref_epochs_long =  pickle.load(f)
    (actData['choiceMI'], actData['choiceMI_max'], actData['choiceMI_max_idx'], actData['choiceMI_pref'], 
     actData['maxMI_epochs'], actData['choiceMI_pref_epochs'],actData['maxMI_epochs_long'], actData['choiceMI_pref_epochs_long']) = \
        (choiceMI, choiceMI_max, choiceMI_max_idx, choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs, maxMI_epochs_long, choiceMI_pref_epochs_long)
    
    with open(load_2P_path+'2P_data_PPC_230130.pkl', 'rb') as f:  # Adding pair_select_idx_sync
        pair_select_sync, select_idx_t, choiceMI, choiceMI_sig_t, choiceMI_max, choiceMI_max_idx,  choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs, maxMI_epochs_long, choiceMI_pref_epochs_long =  pickle.load(f)
    (actData['pair_select_sync'],actData['select_idx_t'],actData['choiceMI'], actData['choiceMI_sig_t'],actData['choiceMI_max'], actData['choiceMI_max_idx'], actData['choiceMI_pref'], 
     actData['maxMI_epochs'], actData['choiceMI_pref_epochs'],actData['maxMI_epochs_long'], actData['choiceMI_pref_epochs_long']) = \
        (pair_select_sync, select_idx_t, choiceMI, choiceMI_max, choiceMI_max_idx, choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs, maxMI_epochs_long, choiceMI_pref_epochs_long)
    '''

    with open(load_2P_path+'2P_data_PPC_230621.pkl', 'rb') as f:  # Adding pair_select_idx_sync with timepoint sigs
        pair_select_sync, select_idx_t, choiceMI, choiceMI_sig_t, choiceMI_prctile, choiceMI_max, choiceMI_max_idx,  choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs, maxMI_epochs_long, choiceMI_pref_epochs_long =  pickle.load(f)
    (actData['pair_select_sync'],actData['select_idx_t'],actData['choiceMI'], actData['choiceMI_sig_t'],actData['choiceMI_prctile'],
     actData['choiceMI_max'], actData['choiceMI_max_idx'], actData['choiceMI_pref'], 
     actData['maxMI_epochs'], actData['choiceMI_pref_epochs'],actData['maxMI_epochs_long'], actData['choiceMI_pref_epochs_long']) = \
        (pair_select_sync, select_idx_t, choiceMI, choiceMI_sig_t, choiceMI_prctile, choiceMI_max, choiceMI_max_idx, choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs, maxMI_epochs_long, choiceMI_pref_epochs_long)
  
    # pairwise corr metrics (Pearson and pairwiseMI)
    with open(load_2P_path+'2P_data_PPC_corrMetrics_230127.pkl','rb') as f:
        (actData['pearsonCorr']) = pickle.load(f)
    return actData

dataset = 'PPC'
select_type = 'MI'
source_type = 'pyr'
n_sessions = 0
n_shuf = 0
sig_thresh=0.95

do_save_MN_DF = 0
do_save_MN_DF_new = 0 # After running matlab code 'save_selectivity_230628.m' to calculate selectivities
do_save_MN_DF_1sess = 0
do_save_tracing_DF = 0
#do_save_typed_DF = 0

do_save_dir_syn_DF = 0
do_save_dir_cn_DF = 0
do_save_pot_dir_cn_DF = 0

do_save_early_middle_late_MN_DF = 0
do_save_early_middle_late_MN_DF_new = 0
do_save_early_middle_late_syn_cn_DFs = 1

do_save_early_series = 0
do_save_middle_series = 0
do_save_late_series = 0
do_save_all_series = 0

#do_save_pot_dir_cn_subtypes_DF = 0

do_save_psp_DF = 0
do_save_syn_DF = 0
do_save_cn_DF = 0

do_save_EIE_triads_DF = 0

do_save_pot_cn_DF = 0

do_save_dir_cn_rate_DF = 0
do_save_conv_triads_DF = 0


do_save_pot_conv_triads_DF = 0
do_save_conv_triads_rate_DF = 0

#do_save_conv_pairs_DF = 0

# analysis dataframe filenames
workingDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/'
DF_types = ['MN_DF', 'MN_DF_new','MN_DF_1sess','tracing_DF', 'typed_DF', 'dir_syn_DF', 
    'dir_cn_DF', 'dir_cn_DF_soma','dir_cn_DF_proximal','dir_cn_DF_distal',
    'dir_cn_DF_apical', 'dir_cn_DF_basal', 
    'psp_DF','syn_DF','cn_DF',
    'pot_dir_cn_DF', 
    'MN_DF_early', 'dir_syn_DF_early', 'dir_cn_DF_early',
    'dir_EIE_cn_DF', 'pot_dir_cn_DF_soma','pot_dir_cn_DF_proximal','pot_dir_cn_DF_distal','pot_dir_cn_DF_apical','pot_dir_cn_DF_basal']
sess_labels = ['early', 'middle', 'late', 'all']
data_DFs = {}
DF_filenames = {}

for DF_type in DF_types:
    #DF_filenames[DF_type] = workingDir+DF_type+'_'+dataset+'_11sess.pkl'
    DF_filenames[DF_type] = workingDir+DF_type+'_'+dataset+'.pkl'
    #for sess_label in sess_labels:
        #DF_filenames[DF_type] = workingDir+DF_type+'_'+dataset+'_11sess.pkl'
        #DF_filenames[DF_type+'_'+sess_label] = workingDir+DF_type+'_'+sess_label+'_'+dataset+'.pkl'
DF_filenames['MN_DF_csv'] = workingDir+'MN_DF_'+dataset+'.csv'

for DF_type in ['early', 'middle', 'late']:
#for DF_type in ['MN_DF_early','MN_DF_middle','MN_DF_late']:
    DF_filenames['MN_DF_'+DF_type] = workingDir+'MN_DF_'+DF_type+'_'+dataset+'.pkl'
    DF_filenames[DF_type+'_csv'] = workingDir+'MN_DF_'+DF_type+'_'+dataset+'.csv'
    DF_filenames['MN_DF_'+DF_type+'_new'] = workingDir+'MN_DF_'+DF_type+'_new_'+dataset+'.pkl'
    DF_filenames['dir_syn_DF_'+DF_type] = workingDir+'dir_syn_DF_'+DF_type+'_'+dataset+'.pkl'
    DF_filenames['dir_cn_DF_'+DF_type] = workingDir+'dir_cn_DF_'+DF_type+'_'+dataset+'.pkl'

'''
conv_pairs_DF_file = workingDir+'MN_conv_pairs_DF_'+dataset+'.pkl'
conv_triads_DF_file = workingDir+'MN_conv_triads_DF_'+dataset+'.pkl'
pot_dir_cn_DF_file = workingDir+'dirMN_pot_cnDF_'+select_type+'_'+dataset+'.pkl'
dir_cn_rate_DF_file = workingDir+'dirMN_cnRateDF_'+select_type+'_'+dataset+'.pkl'

pot_cn_DF_file = workingDir+'pot_cn_DF_'+select_type+'_'+dataset+'.pkl'
pot_conv_triads_DF_file = workingDir+'pot_conv_triads_DF_'+select_type+'_'+dataset+'.pkl'
conv_triads_rate_DF_file = workingDir+'conv_triads_rate_DF_'+select_type+'_'+dataset+'.pkl'
'''
# Load dataset specific parameters (PPC vs V1)
load_2P_path = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/select_metrics/"
if dataset == 'PPC' or dataset == 'PPC_test': # load activity data
    actData = load_PPC_actData(mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213'])
elif dataset == 'V1' or dataset == 'V1_test' or dataset=='V1_sources':
    actData = {}
    blocks = None
    with open('correspondence/EMidFunctMat.pkl', 'rb') as f:  
        EMidFunctMat_df = pickle.load(f)
    actData['EMidFuncMat_df'] = EMidFunctMat_df
shuf_metrics = ['','_cueEarly','_cueLate','_delayEarly','_delayTurn','_turnITI']
shuf_metrics = ['selectivity_ROC' + i for i in shuf_metrics ]

# define selectivity index measures
if dataset == 'V1_test' or dataset == 'V1' or dataset=='V1_sources':
    select_idx = 'oripeaksel'
    selectivity = None
elif dataset == 'PPC_test' or dataset == 'PPC':
    select_idx = 'select_idx_'+select_type
    selectivity = 'selectivity_'+select_type
myDF.load_dataset(dataset)

# Save matched neurons dataframes
if do_save_MN_DF:
    print('Saving neurons dataframe %s' %  DF_filenames['MN_DF'])
    data_DFs['MN_DF'] = myDF.gen_neurons_df(dataset, actData, sig_thresh=sig_thresh, shuf_metrics = [select_idx, selectivity], n_shuf = n_shuf)
    with open(DF_filenames['MN_DF'], 'wb') as f:  
        pickle.dump(data_DFs['MN_DF'], f)
    # also save as csv for matlab code 
    MN_DF.to_csv(DF_filenames['MN_DF_csv'])
else:
    for DF_type in ['MN_DF']:
        with open(DF_filenames[DF_type], 'rb') as f:
            data_DFs[DF_type] = pickle.load(f)

if do_save_MN_DF_new: 
    with open(DF_filenames['MN_DF'], 'rb') as f:  
        MN_DF = pickle.load(f)
    mat_path = (workingDir+'MN_DF_new_select.mat')
    with h5py.File(mat_path, 'r') as f:
        Info_values_final= np.array(f['Info_values_final'])
        Infomax_final = np.array(f['Infomax_final'])
    MN_DF_new = MN_DF
    MN_DF_new['select_idx_MI_new'] = Infomax_final
    MN_DF_new['select_idx_MI_t_new'] =  [Info_values_final[:,i] for i in np.arange(143)]
    data_DFs['MN_DF_new'] = MN_DF_new
    with open(workingDir + 'MN_DF_new_PPC.pkl','wb') as f:
        pickle.dump(MN_DF_new, f)
else:
    for DF_type in ['MN_DF_new']:
        with open(DF_filenames[DF_type], 'rb') as f:
            data_DFs[DF_type] = pickle.load(f)

if do_save_MN_DF_1sess:
    print('Saving neurons dataframe (1 session) %s' %  DF_filenames['MN_DF'])
    data_DFs['MN_DF_1sess'] = myDF.gen_neurons_df(dataset, load_PPC_actData(mySessions = ['LD187_141216']))
    with open(DF_filenames['MN_DF_1sess'], 'wb') as f:  
        pickle.dump(data_DFs['MN_DF_1sess'], f)

# Save tracing progress dataframe
if do_save_tracing_DF:
    print('Saving tracing dataframe %s' % DF_filenames['tracing_DF'])
    data_DFs['tracing_DF'] = myDF.gen_tracing_DF(dataset=dataset)
    with open(DF_filenames['tracing_DF'], 'wb') as f:
        pickle.dump(data_DFs['tracing_DF'], f)

# Save direct synapse dataframes
if do_save_dir_syn_DF:
    print('Saving direct synapse dataframe %s' % DF_filenames['dir_syn_DF'])
    data_DFs['dir_syn_DF']= myDF.gen_synapse_df(data_DFs['MN_DF_new'], data_DFs['MN_DF_new'], actData, dataset=dataset, 
        dir_cn=True,add_dists = True, select_idx=select_idx, selectivity=selectivity)
    with open(DF_filenames['dir_syn_DF'] , 'wb') as f:
        pickle.dump(data_DFs['dir_syn_DF'], f)
else:
    #for DF_type in ['dir_syn_DF', 'pyr_dir_syn_DF', 'nonpyr_dir_syn_DF']:
    with open(DF_filenames['dir_syn_DF'], 'rb') as f:
        data_DFs['dir_syn_DF'] = pickle.load(f)

# Save direct MN cn dataframe
if do_save_dir_cn_DF: 
    print('Saving direct connection dataframe %s' % DF_filenames['dir_cn_DF'])
    data_DFs['dir_cn_DF'] = myDF.gen_cn_DF(data_DFs['MN_DF_new'],data_DFs['dir_syn_DF'], data_DFs['MN_DF_new'], actData, dataset=dataset, dir_cns=True, cable_overlaps=True,
        select_idx=select_idx, selectivity=selectivity)
    with open(DF_filenames['dir_cn_DF'], 'wb') as f:  
        pickle.dump(data_DFs['dir_cn_DF'], f)
    # Save dir cn DFs for particular types of dendrite targets
    '''
    for den_type in ['soma', 'proximal','apical','basal']: # subdivide distal dendrites into apical and basal
        data_DFs['dir_cn_DF_'+den_type] = myDF.gen_cn_DF(data_DFs['MN_DF'],data_DFs['dir_syn_DF'][data_DFs['dir_syn_DF'].den_type==den_type],
            data_DFs['MN_DF'], actData, dataset=dataset, dir_cns=True, cable_overlaps=True,select_idx=select_idx, selectivity=selectivity,
            dendrite_type=den_type)
        print('Saving %s connection dataframe %s' % (den_type, DF_filenames['dir_cn_DF_'+den_type]))
        with open(DF_filenames['dir_cn_DF_'+den_type], 'wb') as f:  
            pickle.dump(data_DFs['dir_cn_DF_'+den_type], f)
    '''
else:
    for DF_type in ['dir_cn_DF']:#, 
        #'dir_cn_DF_proximal', 'dir_cn_DF_distal', 'dir_cn_DF_apical', 'dir_cn_DF_basal']:
        with open(DF_filenames[DF_type], 'rb') as f:
            data_DFs[DF_type] = pickle.load(f)

# Save potential dir cn (includes non-connects)
if do_save_pot_dir_cn_DF:
    data_DFs['pot_dir_cn_DF'] = myDF.gen_pot_cn_DF(data_DFs['MN_DF_new'], data_DFs['MN_DF_new'], data_DFs['dir_syn_DF'], actData = actData, 
        dir_cns = True, select_idx = select_idx, selectivity=selectivity, dataset=dataset,
        min_axon_len=0, min_synout=0, add_select=True, cable_overlaps=True, min_overlap=0)
    print('Saving potential direct connection dataframe %s' % DF_filenames['pot_dir_cn_DF'])
    with open(DF_filenames['pot_dir_cn_DF'], 'wb') as f:
        pickle.dump(data_DFs['pot_dir_cn_DF'], f)

# Save early / middle sessions MN, dir_syn, and dir_cn DFs
def save_other_sessions_DFs(sess_label, actData_custom, DF_type = 'MN_DF'):
    
    if DF_type == "MN_DF":
        print('Saving neurons dataframe (%s sessions) %s' %  (sess_label, DF_filenames['MN_DF_'+sess_label]))
        data_DFs['MN_DF_'+sess_label] = myDF.gen_neurons_df(dataset, actData_custom)
        with open(DF_filenames['MN_DF_'+sess_label], 'wb') as f:  
            pickle.dump(data_DFs['MN_DF_'+sess_label], f)

    if DF_type == "syn_DF": 
        print('Saving direct synapse dataframe (%s sessions) %s' % (sess_label, DF_filenames['dir_syn_DF_'+sess_label]))
        data_DFs['dir_syn_DF_'+sess_label]= myDF.gen_synapse_df(data_DFs['MN_DF_'+sess_label+'_new'], data_DFs['MN_DF_'+sess_label+'_new'], actData_custom, dataset=dataset, 
            dir_cn=True,add_dists = True, select_idx=select_idx, selectivity=selectivity)
        with open(DF_filenames['dir_syn_DF_'+sess_label] , 'wb') as f:
            pickle.dump(data_DFs['dir_syn_DF_'+sess_label], f)

    if DF_type == "cn_DF":
        print('Saving direct connection dataframe (%s sessions) %s' % (sess_label, DF_filenames['dir_cn_DF_'+sess_label]))
        data_DFs['dir_cn_DF_'+sess_label] = myDF.gen_cn_DF(data_DFs['MN_DF_'+sess_label+'_new'], data_DFs['dir_syn_DF_'+sess_label], data_DFs['MN_DF_'+sess_label+'_new'], 
        actData_custom, dataset=dataset, dir_cns=True, cable_overlaps=True,select_idx=select_idx, selectivity=selectivity)
        with open(DF_filenames['dir_cn_DF_'+sess_label], 'wb') as f:  
            pickle.dump(data_DFs['dir_cn_DF_'+sess_label], f)

if do_save_early_middle_late_MN_DF:
    actData_early = load_PPC_actData(mySessions = ['LD187_141208','LD187_141207','LD187_141206'])
    save_other_sessions_DFs('early', actData_early, DF_type = 'MN_DF')
    actData_middle = load_PPC_actData(mySessions = ['LD187_141212','LD187_141211','LD187_141210','LD187_141209'])
    save_other_sessions_DFs('middle', actData_middle, DF_type = 'MN_DF')
    actData_late = load_PPC_actData(mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213'])
    save_other_sessions_DFs('late', actData_late, DF_type = 'MN_DF')
    # Save as .csv for downstream matlab code 
    for DF_type in ['MN_DF_early','MN_DF_middle','MN_DF_late']:
        with open(DF_filenames[DF_type], 'rb') as f:
            data_DFs[DF_type] = pickle.load(f)
        data_DFs[DF_type].to_csv(DF_filenames[DF_type+'_csv'])
else:
    for DF_type in ['MN_DF_early','MN_DF_middle','MN_DF_late']:
        with open(DF_filenames[DF_type], 'rb') as f:
            data_DFs[DF_type] = pickle.load(f)

if do_save_early_middle_late_MN_DF_new:
    for DF_type in ['early','middle','late']:
        with open(DF_filenames['MN_DF_'+DF_type], 'rb') as f:  
            MN_DF = pickle.load(f)
        mat_path = (workingDir+'MN_DF_'+DF_type+'_new_select.mat')
        with h5py.File(mat_path, 'r') as f:
            Info_values_final= np.array(f['Info_values_final'])
            Infomax_final = np.array(f['Infomax_final'])
        MN_DF_new = MN_DF
        MN_DF_new['select_idx_MI_new'] = Infomax_final
        MN_DF_new['select_idx_MI_t_new'] =  [Info_values_final[:,i] for i in np.arange(143)]
        data_DFs['MN_DF_'+DF_type+'_new'] = MN_DF_new
        with open(DF_filenames['MN_DF_'+DF_type+'_new'],'wb') as f:
            pickle.dump(MN_DF_new, f)
else:
    for DF_type in ['early','middle','late']:
        with open(DF_filenames['MN_DF_'+DF_type+'_new'], 'rb') as f:
            data_DFs['MN_DF_'+DF_type+'_new'] = pickle.load(f)

if do_save_early_middle_late_syn_cn_DFs:
    actData_early = load_PPC_actData(mySessions = ['LD187_141208','LD187_141207','LD187_141206'])
    save_other_sessions_DFs('early', actData_early, DF_type = 'syn_DF')
    save_other_sessions_DFs('early', actData_early, DF_type = 'cn_DF')

    actData_middle = load_PPC_actData(mySessions = ['LD187_141212','LD187_141211','LD187_141210','LD187_141209'])
    save_other_sessions_DFs('middle', actData_middle, DF_type = 'syn_DF')
    save_other_sessions_DFs('middle', actData_middle, DF_type = 'cn_DF')

    actData_late = load_PPC_actData(mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213'])
    save_other_sessions_DFs('late', actData_late, DF_type = 'syn_DF')
    save_other_sessions_DFs('late', actData_late, DF_type = 'cn_DF')

'''
if do_save_early_series:
    actData_early = load_PPC_actData(mySessions = ['LD187_141208','LD187_141207','LD187_141206'])
    save_other_sessions_DFs('early', actData_early)

if do_save_middle_series:
    actData_middle = load_PPC_actData(mySessions = ['LD187_141212','LD187_141211','LD187_141210','LD187_141209'])
    save_other_sessions_DFs('middle', actData_middle)

if do_save_late_series:
    actData_late = load_PPC_actData(mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213'])
    save_other_sessions_DFs('late', actData_late)

if do_save_all_series:
    actData_all = load_PPC_actData(mySessions = allSessions)
    save_other_sessions_DFs('all', actData_all)

if do_save_EIE_triads_DF:
    data_DFs['dir_EIE_cn_DF'] = myDF.gen_EIE_triads_DF(data_DFs['pyr_dir_cn_DF'],data_DFs['nonpyr_dir_cn_DF'])
    print('Saving EIE triads dataframe %s' % DF_filenames['dir_EIE_cn_DF'])
    with open(DF_filenames['dir_EIE_cn_DF'], 'wb') as f:
        pickle.dump(data_DFs['dir_EIE_cn_DF'], f) 
'''


# Save PSP dataframes
if do_save_psp_DF:
    data_DFs['psp_DF'] = myDF.gen_psp_df(data_DFs['MN_DF'], dataset=dataset)
    with open(DF_filenames['psp_DF'], 'wb') as f:
        pickle.dump(data_DFs['psp_DF'], f)
else:
    with open(DF_filenames['psp_DF'], 'rb') as f:
        data_DFs['psp_DF'] = pickle.load(f)

# Save synapse dataframes
if do_save_syn_DF:
    add_dists = True
    #if add_dists:
    #    DF_filenames['syn_DF'] = 'local_data/MN_synDF_'+dataset+'_dists.pkl'
    data_DFs['syn_DF'] = myDF.gen_synapse_df(data_DFs['MN_DF'], data_DFs['psp_DF'], actData, dataset=dataset, dir_cn = False, 
        add_dists = add_dists, select_idx = select_idx, selectivity = selectivity)
    print('Saving synapse dataframe %s' % DF_filenames['syn_DF'])   
    with open(DF_filenames['syn_DF'] , 'wb') as f:
        pickle.dump(data_DFs['syn_DF'], f)
        
    '''
    print('Saving pyr synapse dataframe %s' % DF_filenames['pyr_syn_DF'])
    data_DFs['pyr_syn_DF'] = data_DFs['syn_DF'][data_DFs['syn_DF'].source_type=='pyramidal']
    with open(DF_filenames['pyr_syn_DF'] , 'wb') as f:
        pickle.dump(data_DFs['pyr_syn_DF'], f)
    print('Saving nonpyr synapse dataframe %s' % DF_filenames['nonpyr_syn_DF']) 
    data_DFs['nonpyr_syn_DF'] = data_DFs['syn_DF'][data_DFs['syn_DF'].source_type=='non pyramidal']
    with open( DF_filenames['nonpyr_syn_DF'], 'wb') as f:
        pickle.dump(data_DFs['nonpyr_syn_DF'], f)
    '''
else:
    #for DF_type in ['syn_DF', 'pyr_syn_DF', 'nonpyr_syn_DF']:
    with open(DF_filenames['syn_DF'], 'rb') as f:
        data_DFs[DF_type] = pickle.load(f)

# Save cn dataframes
if do_save_cn_DF: 
    print('Saving connection dataframe %s' % DF_filenames['cn_DF'])
    data_DFs['cn_DF']= myDF.gen_cn_DF(data_DFs['MN_DF'], data_DFs['syn_DF'], data_DFs['psp_DF'], actData, select_idx = select_idx, dataset=dataset,
        selectivity = selectivity, cable_overlaps=False)
    with open(DF_filenames['cn_DF'], 'wb') as f:  
        pickle.dump(data_DFs['cn_DF'], f)
else:
    with open(DF_filenames['cn_DF'], 'rb') as f:  
        cn_DF = pickle.load(f)

# Save potential direct cn dataframe


'''
# Save convergent pairs dataframes
if do_save_conv_pairs_DF:
    print('Saving convergent pairs dataframe %s' % conv_pairs_DF_file)
    conv_pairs_DF = myDF.gen_conv_pair_DF(MN_DF, cn_DF, psp_DF, n_shuf=0, dataset='PPC',select_idx=select_idx,selectivity=selectivity)
    with open(conv_pairs_DF_file,'wb') as f:
        pickle.dump(conv_pairs_DF,f)
else:
    with open(conv_pairs_DF_file,'rb') as f:
        conv_pairs_DF = pickle.load(f)
'''

# Save direct cn rate dataframes
if do_save_dir_cn_rate_DF:
    pot_dir_cn_DF = myDF.gen_pot_cn_DF(MN_DF, MN_DF, syn_DF, actData = actData, 
        dir_cns = True, select_idx = select_idx, selectivity=selectivity, dataset=dataset,
        min_axon_len=100, min_synout=2, add_select=True, cable_overlaps=True, min_overlap=0)
    print('Saving potential direct connection dataframe %s' % pot_dir_cn_DF_file)
    with open(pot_dir_cn_DF_file, 'wb') as f:
        pickle.dump(pot_dir_cn_DF, f)
    dir_cn_rate_DF = myDF.get_cn_rate_DF(pot_dir_cn_DF, count_mult=True, n_shuf=n_shuf)
    with open(dir_cn_rate_DF_file, 'wb') as f:
        pickle.dump(dir_cn_rate_DF, f)
    print('Saving direct connection rate dataframe %s' % dir_cn_rate_DF_file)

# Save convergent triads dataframes
if do_save_conv_triads_DF:
    print('Saving convergent triads dataframe %s' % conv_triads_DF_file)
    conv_triads_DF = myDF.gen_conv_triads_DF(cn_DF, MN_DF, actData, select_idx = select_idx, selectivity=selectivity, n_shuf = n_shuf)
    with open(conv_triads_DF_file, 'wb') as f:
        pickle.dump(conv_triads_DF, f)
else:
    with open(conv_triads_DF_file, 'rb') as f:
        conv_triads_DF = pickle.load(f)

# Save convergent triads rates dataframe

if do_save_pot_conv_triads_DF:
    # for now, only EE-I
    pot_cn_DF = pot_cn_DF[pot_cn_DF.target_type == 'non pyramidal']
    pot_cn_DF = pot_cn_DF[pot_cn_DF.source_type == 'pyramidal']

    # for now, only connections with non-zero cable overlap
    #pot_cn_DF = pot_cn_DF[pot_cn_DF.cable_overlap > 0]
    # for now, restrict to max soma dist of 100 um
    max_soma_dist = 200

    pot_conv_triads_DF = myDF.gen_pot_conv_triads_DF(MN_DF, psp_DF, syn_DF, pot_cn_DF, max_soma_dist = max_soma_dist)
    print('Saving potential conv triads dataframe as %s' % pot_conv_triads_DF_file)
    with open(pot_conv_triads_DF_file,'wb') as f:
        pickle.dump(pot_conv_triads_DF, f)
else:
    with open(pot_conv_triads_DF_file,'rb') as f:
        pot_conv_triads_DF = pickle.load(f)
if do_save_conv_triads_rate_DF:
    conv_triads_rate_DF = myDF.get_conv_triads_rate_DF(pot_conv_triads_DF, n_shuf = n_shuf, soma_dist_bins = np.linspace(0,400,15), 
        corr_bins = [-1,.1,1],cable_overlap_bins = np.linspace(0,200,15), datasets = 'PPC', count_mult=False)
    print('Saving conv triads rate dataframe as %s' % conv_triads_rate_DF_file)
    with open(conv_triads_rate_DF_file,'wb') as f:
        pickle.dump(conv_triads_rate_DF, f)
else:
    with open(conv_triads_rate_DF_file,'rb') as f:
        conv_triads_rate_DF = pickle.load(f)



