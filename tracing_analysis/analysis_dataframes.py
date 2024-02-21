# functions for generating various dataframes for analysis
import pymaid
import navis
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import sys
import os
import h5py
import pickle
import networkx as nx
import importlib
import scipy
import ppc_analysis_functions.catmaid_API as cAPI
importlib.reload(cAPI)
from itertools import combinations
import itertools
import logging, sys
logging.disable(sys.maxsize)

my_prx_r = 64# Determine cutoff for soma/proximal connections for dendrite compartments

def load_dataset(dataset, annotations=None):
        if dataset == 'PPC':
            rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                                        api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff', project_id=1)
            if annotations is not None:
                mySkelIDs = pymaid.get_skids_by_annotation(annotations,allow_partial=False)
            else:
                mySkelIDs = pymaid.get_skids_by_annotation("new matched neuron ",allow_partial=True)
            #print(len(mySkelIDs))
        elif dataset == 'PPC_test':
            rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                                        api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff', project_id=1)
            mySkelIDs = np.array([22901,139663,142354])
        elif dataset == 'V1':
            rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                                        api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff', project_id=31)
            if annotations is not None:
                mySkelIDs = pymaid.get_skids_by_annotation(annotations,allow_partial=False)
            else:
                mySkelIDs = pymaid.get_skids_by_annotation("matched neuron ",allow_partial=True)
            #print(len(mySkelIDs))
        elif dataset == 'V1_test':
            rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                                        api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff', project_id=31)
            mySkelIDs = np.array([250832,251169,251080])
        elif dataset == 'V1_sources':
            rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                            api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff', project_id=31)
            mySkelIDs = pymaid.get_skids_by_annotation(['matched neuron ','source neuron'],allow_partial=True)
        else:
            print('ERROR: dataset not recognized')
            mySkelIDs = []
        return mySkelIDs
def avg_lr(row):
    try:
        return (row['Ca_trial_mean_bR']+row['Ca_trial_mean_wL'])/2
    except:
        return np.nan

def find_max_idx(Ca):
    if np.isnan(Ca).any():
        return np.nan
    else:
        return np.nanargmax(Ca)
def gen_neurons_df(dataset, actData, neuron_type=None, shuf_metrics=[], n_shuf=0, sig_thresh=0.9994):
    def init_matched_neuronDf(skelID, labels, actData, isMatched=False, dataset='PPC'):
        myNeuronDf = pd.Series(name=int(skelID))
        myLabels = labels[ labels.skeleton_id == int(skelID) ]
        myNeuron = pymaid.get_neuron(skelID)
        if dataset == 'PPC' or dataset == 'PPC_test':
            myNeuronDf = cAPI.query_cell_info(myNeuronDf,myNeuron,myLabels, parseLayers=False)
            if isMatched:
                mySessions = actData['mySessions']
                '''
                (AUC_fullTrial, AUC_blocks, RL_selectivity_ROC,RL_selectivity_ROC_blocks) = (actData['AUC_fullTrial'], 
                    actData['AUC_blocks'], actData['RL_selectivity_ROC'], actData['RL_selectivity_ROC_blocks'])
                (RL_fullTrial, RL_blocks, RL_selectivity, RL_selectivity_blocks) = (actData['RL_fullTrial'], 
                    actData['RL_blocks'], actData['RL_selectivity'], actData['RL_selectivity_blocks'])
                '''
                #trial_snr = actData['trial_snr']
                
                #(choiceMI, choiceMI_max, choiceMI_max_idx, maxMI_sig, choiceMI_prctile, choiceMI_pref) = \
                #    (actData['choiceMI'], actData['choiceMI_max'], actData['choiceMI_max_idx'], actData['maxMI_sig'],
                #    actData['choiceMI_prctile'], actData['choiceMI_pref'])
                
                (choiceMI, choiceMI_sig_t, choiceMI_prctile, choiceMI_max,  choiceMI_max_idx,  choiceMI_pref, maxMI_epochs, choiceMI_pref_epochs, maxMI_epochs_long, choiceMI_pref_epochs_long) = \
                   (actData['choiceMI'], actData['choiceMI_sig_t'],actData['choiceMI_prctile'],actData['choiceMI_max'], actData['choiceMI_max_idx'], actData['choiceMI_pref'], 
                    actData['maxMI_epochs'], actData['choiceMI_pref_epochs'],actData['maxMI_epochs_long'], actData['choiceMI_pref_epochs_long'])
                select_idx_t = actData['select_idx_t']
                
                myCellID = cAPI.get_cid_from_skelID(skelID)
                myNeuronDf['matched_cell_ID'] = myCellID
                myNeuronDf['corr_quality'] = cAPI.get_corr_quality(skelID)
                cAPI.get_2P_ROIs(myNeuronDf,myCellID,myNeuron,myLabels,mySessions)
                #cAPI.query_selectivity(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,
                #    AUC_fullTrial,AUC_blocks, RL_selectivity_ROC,RL_selectivity_ROC_blocks, select_metric='ROC')
                #cAPI.query_selectivity(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,
                #    RL_fullTrial, RL_blocks, RL_selectivity, RL_selectivity_blocks, select_metric='RL')
                cAPI.query_MI_selectivity(myNeuronDf, myCellID, mySessions, choiceMI_max, 
                    choiceMI_pref, select_metric = 'MI', sig_thresh = sig_thresh)
                cAPI.query_MI_selectivity_t(myNeuronDf, myCellID, mySessions, select_idx_t, 
                    select_metric = 'MI_t', filter=False)
                '''
                cAPI.query_MI_selectivity_t(myNeuronDf, myCellID, mySessions, choiceMI_prctile, 
                    select_metric = 'MI_prctile', filter=False)
                cAPI.query_MI_selectivity_t(myNeuronDf, myCellID, mySessions, select_idx_t, 
                    select_metric = 'MI_t_filt', filter=True, choiceMI_sig_t=choiceMI_sig_t)
                cAPI.query_MI_selectivity_t_prctile(myNeuronDf, myCellID, mySessions, select_idx_t, 
                    select_metric = 'MI_t_filt_95', filter=True, choiceMI_prctile=choiceMI_prctile, thresh = 0.95)
                cAPI.query_MI_selectivity_t_prctile(myNeuronDf, myCellID, mySessions, select_idx_t, 
                    select_metric = 'MI_t_filt_68', filter=True, choiceMI_prctile=choiceMI_prctile, thresh = 0.68)
                for epoch_idx, epoch in enumerate(['ITIbefore','cueEarly','cueLate','delay','turn','ITI']):
                    cAPI.query_MI_selectivity(myNeuronDf, myCellID, mySessions, maxMI_epochs, 
                        choiceMI_pref_epochs, select_metric = 'MI_'+epoch, sig_thresh = sig_thresh, column = epoch_idx)
                for epoch_long_idx, epoch_long in enumerate(['cueAll','delayAll','turnAll']):
                    cAPI.query_MI_selectivity(myNeuronDf, myCellID, mySessions, maxMI_epochs_long, 
                        choiceMI_pref_epochs_long, select_metric = 'MI_'+epoch_long, sig_thresh = sig_thresh, column = epoch_long_idx)
                '''
                
                #cAPI.query_psp_tracing(myNeuronDf,myCellID,myNeuron,myLabels)
                #cAPI.query_trial_snr(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,trial_snr)
                # Add timing measures
                #cAPI.query_t_peak(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,actData['tCOM'],label='tCOM')
                
                cAPI.query_t_peak(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,actData['choiceMI_max_idx'],label='choiceMI_max_idx') 
                cAPI.query_trialAvgActivity(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,actData['Ca_trial_means']) 
                cAPI.query_trialAvgMI(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,actData['choiceMI'])
            cAPI.query_axon(myNeuronDf,myCellID,myNeuron,myLabels)
            #cAPI.query_BS_inputs(myNeuronDf,myCellID,myNeuron,myLabels)
            #cAPI.query_CC_inputs(myNeuronDf,myCellID,myNeuron,myLabels)
        elif dataset == 'V1' or dataset == 'V1_test' or dataset == 'V1_sources':
            myNeuronDf = cAPI.query_cell_info(myNeuronDf,myNeuron,myLabels,parseLayers=False)
            if isMatched:
                myCellID = cAPI.get_cid_from_skelID(skelID, annotTag='matched neuron')
                myNeuronDf['matched_cell_ID'] = myCellID
                query_V1_selectivity(myNeuronDf,myCellID,myNeuron,myLabels,actData['EMidFuncMat_df'])
                #cAPI.query_psp_tracing(myNeuronDf,myCellID,myNeuron,myLabels)
            cAPI.query_axon(myNeuronDf,myCellID,myNeuron,myLabels)
        return(myNeuronDf)
    def query_V1_selectivity(myNeuronDf,myCellID,myNeuron,myLabels,V1_phys):
        EMidFunctMat_cols = ['physIDs','dirpeaksel','oripeaksel','posXpeaksel','posYpeaksel','sfpeaksel',
                        'tfpeaksel','speedpeaksel','alphasel','oritunsel']
        for feature in EMidFunctMat_cols:
            try:   
                myNeuronDf[feature] = V1_phys[V1_phys.matched_cell_ID==myNeuronDf.matched_cell_ID][feature].values[0]
            except:
                myNeuronDf[feature] = np.nan

    mySkelIDs = load_dataset(dataset)
    pymaid.clear_cache()
    labels = pymaid.get_label_list()

    # Make dataframe of tracing data for Matched Neurons 
    tracingDf = pd.DataFrame([],dtype=int)
    t = time.time()
    for i, skelID in enumerate(mySkelIDs):
        print('assembling neuron %i of %i' % (i+1, len(mySkelIDs)))
        myNeuronDf = init_matched_neuronDf(skelID, labels, actData, isMatched=True, dataset=dataset)
        tracingDf = tracingDf.append(myNeuronDf, ignore_index=False)
    #tracingDf = tracingDf.astype({"skeleton_id": int})
    print('Assembling dataframe took %s seconds' % str(time.time()-t)) 
    if dataset is not 'V1_sources':
        tracingDf.sort_values('matched_cell_ID', ascending=True, inplace=True) 

    # Analyze pyramidal network with traced PSPs
    if dataset == 'PPC' or dataset == 'PPC_test':
        tracingDf = tracingDf[tracingDf.corr_quality.isin(['good','okay'])]
    myNeuronsDf = tracingDf
    if neuron_type is not None:
        myNeuronsDf = myNeuronsDf[myNeuronsDf.type == neuron_type]
    #mySources = mySources[mySources.num_synout>0]

    if shuf_metrics is not None and n_shuf > 0:
        for i in range(n_shuf):
            shuf_metrics_shuf = [a+'_shuffle_'+str(i) for a in shuf_metrics]
            myNeuronsDf[shuf_metrics_shuf] = pd.DataFrame(myNeuronsDf.sample(frac=1,
                replace=False, axis='rows')[shuf_metrics].values, index = myNeuronsDf.index)
    return myNeuronsDf

def gen_collats_df(neurons_df, collats_cols = ['collat_syn_density', 'collat_syn_count', 'collat_lengths'], neuron_cols = ['select_idx_MI_abs', 'trial_snr_max','selectivity_MI']):
    collats_df = pd.DataFrame()
    neurons_df.dropna(axis = 0, subset = collats_cols, inplace=True)
    for col in collats_cols:
        collats_df[col] = [item for sublist in neurons_df[col].values for item in sublist] 
    for col in neuron_cols:
        neurons_df[col] = neurons_df.apply (lambda row: [row[col] for i in row[collats_cols[0]].values], axis=1)
        collats_df[col] = [item for sublist in neurons_df[col].values for item in sublist] 
    return collats_df

def gen_tracing_DF(dataset='PPC'):
    def init_tracing_DF(skelID, PSP_df, post_cn_details, labels, annots):
        myNeuronDf = pd.Series(name=int(skelID))
        myLabels = labels[ labels.skeleton_id == int(skelID) ]
        myNeuron = pymaid.get_neuron(skelID)
        myNeuronDf = cAPI.query_cell_info(myNeuronDf,myNeuron,myLabels, parseLayers=False)
        myCellID = cAPI.get_cid_from_skelID(skelID)
        myNeuronDf['matched_cell_ID'] = myCellID
        cAPI.query_psp_tracing(myNeuronDf,myCellID,myNeuron,post_cn_details,PSP_df,annots)
        return(myNeuronDf)
    mySkelIDs = load_dataset(dataset)
    pymaid.clear_cache()
    PSpartners = pymaid.get_partners(mySkelIDs,directions=['outgoing'],threshold=0,min_size=0)
    PSP_df = pymaid.get_neurons(PSpartners)
    post_cns = pymaid.get_connectors(mySkelIDs,relation_type='presynaptic_to').connector_id.values
    post_cn_details = pymaid.get_connector_details(post_cns)
    annots = pymaid.get_annotations(PSP_df)
    labels = pymaid.get_label_list()
    # Make dataframe of tracing data for Matched Neurons 
    tracing_DF = pd.DataFrame([],dtype=int)
    t = time.time()
    for i, skelID in enumerate(mySkelIDs):
        print('assembling neuron %i of %i' % (i+1, len(mySkelIDs)))
        myNeuronDf = init_tracing_DF(skelID, PSP_df, post_cn_details, labels, annots)
        tracing_DF = tracing_DF.append(myNeuronDf, ignore_index=False)
    print('Assembling dataframe took %s seconds' % str(time.time()-t)) 

    return tracing_DF

def gen_psp_df(neurons_DF, dataset='PPC'):
    def init_psp_neuronDf(skelID, labels, isMatched=False, dataset='PPC'):
        myNeuronDf = pd.Series(name=int(skelID))
        myLabels = labels[ labels.skeleton_id == int(skelID)]
        myNeuron = pymaid.get_neuron(skelID)
        if dataset == 'PPC' or dataset == 'PPC_test':
            myNeuronDf = cAPI.query_cell_info(myNeuronDf,myNeuron,myLabels, parseLayers=False)
        elif dataset == 'V1' or dataset == 'V1_test':
            myNeuronDf = cAPI.query_cell_info(myNeuronDf,myNeuron,myLabels, parseLayers=False)
        return(myNeuronDf)
    print('Seeding network from %i matched neurons' % len(neurons_DF))
    matchedNeuronList = pymaid.get_neuron(neurons_DF.skeleton_id.values.astype(int))
    PSpartnersList = pymaid.get_partners(matchedNeuronList,directions=['outgoing'],threshold=0,min_size=0)
    print('%i total partners' % len(PSpartnersList)) 

    psp_df = pd.DataFrame()
    mySkelIDs = PSpartnersList.skeleton_id[:]
    pymaid.clear_cache()
    labels = pymaid.get_label_list()
    t = time.time()
    for i,skelID in enumerate(mySkelIDs):
        print('%i of %i PSPs' % (i+1, len(mySkelIDs)))
        my_df = init_psp_neuronDf(skelID, labels, isMatched=False, dataset=dataset)
        psp_df = psp_df.append(my_df, ignore_index=True)
    psp_df = psp_df.astype({"skeleton_id": int}) 

    # classify tracing status of ps partners
    ps_annot = pymaid.get_annotations(psp_df.skeleton_id.values)
    psp_df = cAPI.add_tracing_status(psp_df, ps_annot)
    print('Assembling PSP dataframe took %s seconds' % str(time.time()-t))
    return psp_df

def gen_all_typed_neurons_df(dataset, annotations=['pyramidal','non pyramidal']):
    def init_psp_neuronDf(skelID, labels, isMatched=False, dataset='PPC'):
        myNeuronDf = pd.Series(name=int(skelID))
        myLabels = labels[ labels.skeleton_id == int(skelID) ]
        myNeuron = pymaid.get_neuron(skelID)
        if dataset == 'PPC' or dataset == 'PPC_test':
            myNeuronDf = cAPI.query_cell_info(myNeuronDf,myNeuron,myLabels, parseLayers=False)
        elif dataset == 'V1' or dataset == 'V1_test':
            myNeuronDf = cAPI.query_cell_info(myNeuronDf,myNeuron,myLabels, parseLayers=False)
        return(myNeuronDf)  
    typed_neuron_df = pd.DataFrame()
    mySkelIDs = load_dataset(dataset, annotations=annotations)
    pymaid.clear_cache()
    labels = pymaid.get_label_list()
    t = time.time()
    for skelID in mySkelIDs:
        my_df = init_psp_neuronDf(skelID, labels, isMatched=False, dataset=dataset)
        typed_neuron_df = typed_neuron_df.append(my_df, ignore_index=True)
    print(typed_neuron_df.columns)
    typed_neuron_df = typed_neuron_df.astype({"skeleton_id": int}) 
    print('Assembling typed neurons dataframe took %s seconds' % str(time.time()-t))
    return typed_neuron_df

def gen_synapse_df(mySources, myTargets, actData, dataset ='PPC', select_idx='select_idx_ROC', den_type = 'dendrite',
        selectivity='selectivity_ROC', blocks=None, n_shuf=0, dir_cn = True, add_dists = False):
    def add_syn_PSsoma_pathdist(PSP_skelID, connector_ID):
        try:
            myNeuron = pymaid.get_neuron(PSP_skelID) #synDF_PPC_test.target.values[0])
            myPSPnode = pymaid.get_connector_details(connector_ID).postsynaptic_to_node.values[0][0]# synDF_PPC_test.connector_id.values[0]).postsynaptic_to_node.values[0][0]
            if myNeuron.soma is not None:
                return navis.dist_between(myNeuron, myPSPnode, myNeuron.soma[0])/1000
            else:
                #print('WARNING: neuron %i has no soma' % int(myNeuron.skeleton_id))
                return np.nan
        except:
            print('WARNING: error in add_synPSsoma_pathdist(%i, %i)' % (PSP_skelID, connector_ID))
            return np.nan
    def add_syn_to_soma_pathdist(skelID, connector_ID):
        myNeuron = pymaid.get_neuron(skelID) #synDF_PPC_test.target.values[0])
        myNode = pymaid.get_connector_details(connector_ID).presynaptic_to_node.values[0]# synDF_PPC_test.connector_id.values[0]).postsynaptic_to_node.values[0][0]
        if myNeuron.soma is not None:
            return navis.dist_between(myNeuron, myNode, myNeuron.soma[0])/1000
        else:
            #print('WARNING: neuron %i has no soma' % int(myNeuron.skeleton_id))
            return np.nan
    mySkelIDs = load_dataset(dataset)
    pymaid.clear_cache()
    labels = pymaid.get_label_list()
    t = time.time()
    syn_DF = cAPI.get_synapses_between(mySources,myTargets)
    if len(syn_DF) == 0:
        return pd.DataFrame()
    syn_DF['is_checked'] = syn_DF.apply(lambda row: cAPI.is_syn_checked(row['connector_id']), axis=1)
    syn_DF['log_psd_area'] = syn_DF.apply (lambda row: np.log10(row['psd_area']), axis=1)
    syn_DF['source_type'] = syn_DF.apply (lambda row: add_source_type(row, mySources), axis=1)
    syn_DF['target_type'] = syn_DF.apply (lambda row: add_target_type(row, myTargets), axis=1)
    syn_DF['cn_type'] = syn_DF.apply (lambda row: add_cn_type(row['source_type'], row['target_type']), axis =1)
    syn_DF['source_select_idx'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx='select_idx_MI'), axis=1)
    syn_DF['source_select_idx_new'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx='select_idx_MI_new'), axis=1)
    #syn_DF['source_select_idx_std'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx=select_idx+'_std'), axis=1)
    #syn_DF['source_select_idx_stdmean'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx=select_idx+'_stdmean'), axis=1)
    syn_DF['source_select_idx_t'] = syn_DF.apply (lambda row: add_source_select(row, mySources,select_idx='select_idx_MI_t'), axis=1)
    syn_DF['source_select_idx_t_new'] = syn_DF.apply (lambda row: add_source_select(row, mySources,select_idx='select_idx_MI_t_new'), axis=1)
    #syn_DF['source_num_active_sessions'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx='num_active_sessions'), axis=1)
    #syn_DF['trial_snr'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx='trial_snr_max'), axis=1)
    #syn_DF['source_selectivity'] = syn_DF.apply (lambda row: add_source_select_class(row, mySources, selectivity=selectivity), axis=1)
    print('Collecting synapses took %s seconds' % str(time.time()-t))
    # Add distance from synapse to soma along dendrite
    if add_dists:
        t = time.time()
        print('Adding synapse to soma path distances')
        syn_DF['syn_PSsoma_pathdist'] = syn_DF.apply (lambda row: add_syn_PSsoma_pathdist(row['target'],row['connector_id']), axis=1)
        syn_DF['syn_soma_pathdist'] = syn_DF.apply (lambda row: add_syn_to_soma_pathdist(row['source'],row['connector_id']), axis=1)  
        syn_DF['source_loc'] = syn_DF.apply (lambda row: add_source_loc(row, mySources), axis=1)  
        syn_DF['den_comp'] = syn_DF.apply (lambda row: classify_dendrite_target_2way(row, labels,proximal_r=my_prx_r), axis=1) 
        syn_DF['den_type'] = syn_DF.apply (lambda row: classify_dendrite_target(row,labels, proximal_r=my_prx_r), axis=1)  
         
        print('Calculating path distances synapses took %s seconds' % str(time.time()-t))    
    if dir_cn:
        #syn_DF = cAPI.add_axon_type(syn_DF)
        syn_DF['target_select_idx'] = syn_DF.apply (lambda row: add_target_select(row, myTargets, select_idx='select_idx_MI'), axis=1)
        syn_DF['target_select_idx_new'] = syn_DF.apply (lambda row: add_target_select(row, myTargets, select_idx='select_idx_MI_new'), axis=1)
        
        #syn_DF['target_selectivity'] = syn_DF.apply (lambda row: add_target_select_class(row, myTargets, selectivity=selectivity), axis=1)
        syn_DF['target_select_idx_t'] = syn_DF.apply (lambda row: add_target_select(row, myTargets,select_idx='select_idx_MI_t'), axis=1)
        syn_DF['target_select_idx_t_new'] = syn_DF.apply (lambda row: add_target_select(row, myTargets,select_idx='select_idx_MI_t_new'), axis=1)

        #syn_DF['RL_diff'] =  syn_DF.apply (lambda row: add_RL_diff(row, select_idx), axis=1)
        #mySynapse_df['RL_sim'] =  mySynapse_df.apply (lambda row: add_RL_sim(row), axis=1)
        if dataset=='PPC' or dataset=='PPC_test':
            #mySynapse_df['pair_selectivity'] = mySynapse_df.apply (lambda row: add_RL_pair_types(row, mySources), axis=1)
            # Add pair selectivity
            #syn_DF['target_select_idx_std'] = syn_DF.apply (lambda row: add_target_select(row, myTargets, select_idx=select_idx+'_std'), axis=1)
            #syn_DF['source_select_idx_std'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx=select_idx+'_std'), axis=1)
            #syn_DF['source_select_idx_stdmean'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx=select_idx+'_stdmean'), axis=1)
            #syn_DF['target_select_idx_stdmean'] = syn_DF.apply (lambda row: add_target_select(row, myTargets, select_idx=select_idx+'_stdmean'), axis=1)
            syn_DF['pair_select_idx'] =  syn_DF.apply (lambda row: add_pair_select_idx(row['source_select_idx'], row['target_select_idx']), axis=1)
            syn_DF['pair_select_idx_new'] =  syn_DF.apply (lambda row: add_pair_select_idx(row['source_select_idx_new'], row['target_select_idx_new']), axis=1)
            
            #syn_DF['pair_select_idx_std'] = syn_DF.apply (lambda row: add_pair_select_idx_stdmean(row['source_select_idx'], row['target_select_idx'], 
            #    row['source_select_idx_std'], row['target_select_idx_std']), axis=1)
            #syn_DF['pair_select_idx_stdmean'] = syn_DF.apply (lambda row: add_pair_select_idx_stdmean(row['source_select_idx'], row['target_select_idx'], 
            #    row['source_select_idx_stdmean'], row['target_select_idx_stdmean']), axis=1)
            syn_DF['pair_select_idx_t'] =  syn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t'], row['target_select_idx_t'], type='t'), axis=1)
            syn_DF['pair_select_idx_t_new'] =  syn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_new'], row['target_select_idx_t_new'], type='t'), axis=1)
            
            syn_DF['pair_select_idx_tmax'] = syn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t'], row['target_select_idx_t'], type='max'), axis=1)
            syn_DF['pair_select_idx_tmax_new'] = syn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_new'], row['target_select_idx_t_new'], type='max'), axis=1)
            
            syn_DF['pair_select_idx_tavg'] = syn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t'], row['target_select_idx_t'], type = 'avg'), axis=1)
            syn_DF['pair_select_idx_tavg_new'] = syn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_new'], row['target_select_idx_t_new'], type = 'avg'), axis=1)
            
            #syn_DF['pair_avg_select_idx'] =  syn_DF.apply (lambda row: add_avg_pair_select_idx(row['source_select_idx_t'], row['target_select_idx_t']), axis=1)
            #syn_DF['pair_selectivity'] = syn_DF.apply (lambda row: add_RL_pair_types_opp(row, mySources), axis=1)
            #syn_DF['source_t_peakMI'] = syn_DF.apply (lambda row: add_source_select(row, mySources,select_idx='choiceMI_max_idx'), axis=1)
            #syn_DF['target_t_peakMI'] = syn_DF.apply (lambda row: add_target_select(row, myTargets,select_idx='choiceMI_max_idx'), axis=1)
            #syn_DF['t_peak_diff'] = syn_DF.apply (lambda row: row['target_t_peakMI']-row['source_t_peakMI'], axis=1)
            #syn_DF['source_tCOM'] = syn_DF.apply (lambda row: add_source_select(row, mySources,select_idx='tCOM'), axis=1)
            #syn_DF['target_tCOM'] = syn_DF.apply (lambda row: add_target_select(row, myTargets,select_idx='tCOM'), axis=1)
            #syn_DF['tCOM_diff'] = syn_DF['target_tCOM']-syn_DF['source_tCOM']
            syn_DF['source_epoch'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx='choiceMI_max_idx_epoch'), axis=1)
            syn_DF['target_epoch'] = syn_DF.apply (lambda row: add_target_select(row, myTargets, select_idx='choiceMI_max_idx_epoch'), axis=1)
            syn_DF['pair_epoch'] = syn_DF.apply (lambda row: compare_epochs(row['source_epoch'], row['target_epoch']), axis=1)

            # Add activity correlations
            syn_DF['corr_trial_avg'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['corr_trial_avg'], actData['mySessions'], mySources), axis=1)
            syn_DF['corr_raw'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['corr_raw'], actData['mySessions'], mySources), axis=1)
            syn_DF['corr_residual'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['corr_residual'], actData['mySessions'], mySources), axis=1)

            #syn_DF['pairMI_lr'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pairMI'], actData['mySessions'], mySources, field='lr'), axis=1)
            #syn_DF['pairMI_trialResidual'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pairMI'], actData['mySessions'], mySources, field='trialResidual'), axis=1)
                                
            syn_DF['pearsonCorr_all'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_all'), axis=1)
            syn_DF['pearsonCorr_lr'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_Ca_lr'), axis=1)
            syn_DF['pearsonCorr_trialMean'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_Ca_trialMean'), axis=1)
            syn_DF['pearsonCorr_trialResidual'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_Ca_residual'), axis=1)
           
           # More activity corrs 220802
            syn_DF['pearsonCorr_trialMean_all'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_trialMean_all'), axis=1)
            syn_DF['pearsonCorr_trialResidual_bwavg'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_trialResidual_bwavg'), axis=1)
            syn_DF['pearsonCorr_bw_diff'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_bw_diff'), axis=1)
            syn_DF['pearsonCorr_timeMean'] = syn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], mySources, field='corr_timeMean'), axis=1)
           
        elif dataset=='V1' or dataset=='V1_test':
            #syn_DF['pair_select_idx'] =  syn_DF.apply (lambda row: add_RL_diff(row, select_idx='oripeaksel'), axis=1)
            syn_DF['source_oritunsel'] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx='oritunsel'), axis=1)
            syn_DF['target_oritunsel'] = syn_DF.apply (lambda row: add_target_select(row, myTargets, select_idx='oritunsel'), axis=1)
            syn_DF['pair_select_idx'] =  syn_DF.apply (lambda row: add_ori_pair_idx(row, avg=None), axis=1)
            syn_DF['oripeakdiff'] =  syn_DF.apply (lambda row: add_ori_pair_idx_old(row), axis=1)
            syn_DF['pair_selectivity'] = syn_DF.apply (lambda row: add_ori_pair_types(row, mySources), axis=1)
        # blockwise selectivity
        if blocks is not None:
            for block in blocks:
                syn_DF['source_select_idx_'+block] = syn_DF.apply (lambda row: add_source_select(row, mySources, select_idx=select_idx+'_'+block), axis=1)
                syn_DF['target_select_idx_'+block] = syn_DF.apply (lambda row: add_target_select(row, myTargets, select_idx=select_idx+'_'+block), axis=1)
                syn_DF['pair_select_idx_'+block] = syn_DF.apply (lambda row: add_pair_select_idx(row['source_select_idx_'+block], row['target_select_idx_'+block]), axis=1)
                if dataset=='PPC' or dataset=='PPC_test':
                    syn_DF['source_selectivity_'+block] = syn_DF.apply (lambda row: add_source_select_class(row, mySources,selectivity=selectivity+'_'+block), axis=1)
                    syn_DF['target_selectivity_'+block] = syn_DF.apply (lambda row: add_target_select_class(row, myTargets,selectivity=selectivity+'_'+block), axis=1)
                    syn_DF['pair_selectivity_'+block] = syn_DF.apply (lambda row: calc_pair_selectivity(row['source_selectivity_'+block],
                            row['target_selectivity_'+block]), axis = 1)
        if n_shuf > 0:
            for i in range(n_shuf):
                print('shuffle %i' % i)
                si = str(i)
                syn_DF['source_selectivity_shuffle_'+si] = syn_DF.apply (lambda row: add_source_select_class(row, mySources, selectivity=selectivity+'_shuffle_'+si), axis=1)
                syn_DF['target_selectivity_shuffle_'+si] = syn_DF.apply (lambda row: add_target_select_class(row, myTargets, selectivity=selectivity+'_shuffle_'+si), axis=1)
                syn_DF['pair_selectivity_shuffle_'+si] = syn_DF.apply (lambda row: calc_pair_selectivity(row['source_selectivity_shuffle_'+si], row['target_selectivity_shuffle_'+si]), axis=1)
                # blockwise selectivity
                if blocks is not None:
                    for block in blocks:
                        source_sel = selectivity+'_'+block+'_shuffle_'+si
                        syn_DF[source_sel] = syn_DF.apply (lambda row: add_source_select_class(row, mySources,selectivity=source_sel), axis=1)
                        target_sel = selectivity+'_'+block+'_shuffle_'+si
                        syn_DF[target_sel] = syn_DF.apply (lambda row: add_target_select_class(row, myTargets,selectivity=target_sel), axis=1)
                        pair_sel = 'pair_selectivity_'+block+'_shuffle_'+si
                        syn_DF[pair_sel] = syn_DF.apply (lambda row: calc_pair_selectivity(row[source_sel],row[target_sel]), axis = 1)
    return syn_DF

def gen_cn_DF(neurons_DF, synapse_DF, psp_DF, actData, select_idx = 'select_idx_MI', selectivity='RL_selectivity_MI',
        n_shuf = 0, add_MN_ids=False, blocks=None, dataset='PPC', cable_overlaps=False, dendrite_type = 'dendrite', add_act_corrs=True, dir_cns = False):
    def cn_df_from_syn_df(synapse_DF, neurons_DF, select_idx='select_idx_MI', selectivity='select_idx_MI'):
        def DiGraph_from_syn(synapse_DF):
            DiG = nx.from_pandas_edgelist(synapse_DF.groupby(['source', 'target'], as_index=False).agg('sum'),
                edge_attr=['psd_area','count'],create_using=nx.DiGraph)
            return DiG
        if len(synapse_DF) > 0:
            DiG =DiGraph_from_syn(synapse_DF)
            cn_DF = nx.to_pandas_edgelist(DiG)
            cn_DF.rename(columns = {'psd_area':'total_psd_area','count':'syn_count'}, inplace = True)
            return cn_DF     
        else:
            return None

    labels = pymaid.get_label_list()
    # restrict sources and targets to those in MN and PSP lists 210921
    synapse_DF = synapse_DF[synapse_DF.source.isin(neurons_DF.skeleton_id)]
    synapse_DF = synapse_DF[synapse_DF.target.isin(psp_DF.skeleton_id)]
    cn_DF = cn_df_from_syn_df(synapse_DF, neurons_DF, select_idx=select_idx, selectivity=selectivity)  
    if cn_DF is None:
        return pd.DataFrame()
    # Add more fields  
    cn_DF['source_type'] = cn_DF.apply (lambda row: add_source_type(row, neurons_DF), axis=1)
    cn_DF['target_type'] = cn_DF.apply (lambda row: add_target_type(row, psp_DF), axis=1)
    cn_DF['cn_type'] = cn_DF.apply (lambda row: add_cn_type(row['source_type'], row['target_type']), axis =1)
    cn_DF['psd_areas'] = cn_DF.apply (lambda row: add_psd_areas(row, synapse_DF), axis=1)
    cn_DF['avg_psd_area'] = cn_DF.apply (lambda row: add_avg_psd_area(row, synapse_DF), axis=1)
    cn_DF['log_avg_psd_area'] = cn_DF.apply (lambda row: np.log10(row['avg_psd_area']), axis=1)
    cn_DF['log_total_psd_area'] = cn_DF.apply (lambda row: np.log10(row['total_psd_area']), axis=1)
    cn_DF['cn_ids'] = cn_DF.apply (lambda row: add_connector_ids(row, synapse_DF), axis=1)
    cn_DF = add_select_to_cnDF(cn_DF, neurons_DF, psp_DF, actData, synapse_DF = synapse_DF, cable_overlaps=cable_overlaps, dendrite_type = dendrite_type,dir_cns = dir_cns, select_idx = select_idx, selectivity=selectivity, dataset=dataset,add_act_corrs = add_act_corrs)
    return cn_DF

def gen_conv_pair_DF(neurons_DF, cn_DF, psp_DF, n_shuf=0, dataset='PPC',select_idx='select_idx_ROC',selectivity='selectivity'):
    from scipy.spatial import distance as dist
    def source_target_labels(myNeuronsDf, field = 'selectivity_ROC'):
        skids = myNeuronsDf[field].values
        source_mat = np.tile(skids,(len(skids),1))
        target_mat = np.tile(skids,(len(skids),1)).transpose()
        source_cond = dist.squareform(source_mat, checks=False)
        target_cond = dist.squareform(target_mat, checks=False)
        return (source_cond, target_cond)
    def pair_labels(myNeuronsDf):
        #skids = myNeuronsDf.matched_cell_ID.values.astype(int)
        skids = myNeuronsDf.skeleton_id.values
        source_mat = np.tile(skids,(len(skids),1))
        target_mat = np.tile(skids,(len(skids),1)).transpose()
        source_cond = dist.squareform(source_mat, checks=False)
        target_cond = dist.squareform(target_mat, checks=False)
        return (source_cond, target_cond)
    def calc_conv_pairs(mySources, myCN_df, n_shuf=n_shuf, select_idx='select_idx_ROC',selectivity='selectivity'):
        def pair_diff(myNeuronsDf, col='select_idx_ROC'):
            cols = myNeuronsDf[col].values.astype(float)
            source_mat = np.tile(cols,(len(cols),1))
            target_mat = np.tile(cols,(len(cols),1)).transpose()
            diff_mat = np.absolute(source_mat - target_mat)
            print(select_idx)
            if select_idx == 'oripeaksel':
                diff_mat = np.minimum(diff_mat, 180-diff_mat)
            elif select_idx == 'dirpeaksel':
                diff_mat = np.minimum(diff_mat, 360-diff_mat)
            diff_mat = dist.squareform(diff_mat, checks=False)
            return (diff_mat)
        def pair_diff_blocks(myNeuronsDf, col='select_idx_blocks'):
            # expect col to have 5-len np array
            diff_mats = dict()
            block_labels = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
            for i,block in enumerate(block_labels):
                cols = np.array([j[i] for j in myNeuronsDf[col].values],dtype=float)
                source_mat = np.tile(cols,(len(cols),1))
                target_mat = np.tile(cols,(len(cols),1)).transpose()
                diff_mat = np.absolute(source_mat - target_mat)
                diff_mat = dist.squareform(diff_mat, checks=False)
                diff_mats[block]=diff_mat
            return (diff_mats)
        def calc_RL_pair_types(myNeuronsDf,selectivity_type = 'selectivity'):
            num_cells = len(myNeuronsDf)
            RL_pair_types =  [[''] * num_cells for i in range(num_cells)]
            RL_pair_types_detail =  [[''] * num_cells for i in range(num_cells)]

            for idx_1,RL_1 in enumerate(myNeuronsDf[selectivity_type]):
                for idx_2, RL_2 in enumerate(myNeuronsDf[selectivity_type]):
                    if RL_1 != 'Non' and RL_1 != 'Mixed' and RL_2 != 'Non' and RL_2 !='Mixed':
                        if RL_1 == RL_2:
                            RL_pair_types[idx_1][idx_2] = 'Same'
                            if RL_1 == 'Left':
                                RL_pair_types_detail[idx_1][idx_2] = 'Left - Left'
                            if RL_1 == 'Right':
                                RL_pair_types_detail[idx_1][idx_2] = 'Right - Right'
                        else:
                            RL_pair_types[idx_1][idx_2] = 'Different'
                            RL_pair_types_detail[idx_1][idx_2] = 'Right - Left'
                    elif RL_1 == 'Non' and RL_2 == 'Non':
                        RL_pair_types[idx_1][idx_2] = 'Different'
                        RL_pair_types_detail[idx_1][idx_2] = 'Non - Non'
                    else:
                        RL_pair_types[idx_1][idx_2] = 'Different'
                        if RL_1 == 'Right' or RL_2 == 'Right':
                            RL_pair_types_detail[idx_1][idx_2] = 'Non - Right'
                        else:
                            RL_pair_types_detail[idx_1][idx_2] = 'Non - Left'
            return dist.squareform(RL_pair_types,checks=False), dist.squareform(RL_pair_types_detail,checks=False)
        def calc_pair_metrics_shuf(mySources, myCN_df, select_idx = 'select_idx_ROC',
                selectivity = 'selectivity_ROC', shuf_metric='selectivity_ROC', n_shuf=0):   
            # calculate activity and connectivity similarity metrics
            select_idx_diff = pair_diff(mySources, col=select_idx)

            #myDiff_ROC = pair_diff(mySources, col='select_idx_ROC')
            #myDiff_ROC_blocks = pair_diff_blocks(mySources, col='select_idx_ROC_blocks')

            (avg_conv_psd_area_sq, avg_conv_psd_area) = cAPI.get_cn_sim(mySources.skeleton_id, 
                synapse_df=myCN_df, metric='avg_conv_psd_area',weight='avg_psd_area')
            avg_conv_psd_area[avg_conv_psd_area==0]=np.nan

            # neuron labels
            (source_labels, target_labels) = pair_labels(mySources)

            # calc activity correlations
            #(myCorr_trial_avg_sq, myCorr_trial_avg, sessions_overlap) = cAPI.calc_act_corr(mySources, corr_trial_avg, mySessions)

            pair_types = dict()
            pair_types['avg_conv_psd_area'] = avg_conv_psd_area
            pair_types['source'] = source_labels
            pair_types['target'] = target_labels
            pair_types['select_idx_diff'] = select_idx_diff
         
            # calc metrics for data
            if selectivity is not None:
                (source_selectivity, target_selectivity) = source_target_labels(mySources, field=selectivity)
                pair_types['source_selectivity'] = source_selectivity
                pair_types['target_selectivity'] = target_selectivity
                # generate pair metrics
                #(my_RL_pair_types, my_RL_pair_types_detail) = calc_RL_pair_types(mySources,  selectivity_type='RL_selectivity')
                #pair_types['pair_selectivity'] = my_RL_pair_types
                #pair_types['pair_selectivity_detail'] = my_RL_pair_types_detail
                (pair_selectivity, pair_selectivity_detail) = calc_RL_pair_types(mySources, selectivity_type=selectivity)
                pair_types['pair_selectivity'] = pair_selectivity
                pair_types['pair_selectivity_detail'] = pair_selectivity_detail

            if dataset=='PPC' or dataset=='PPC_test':
                select_idx_diff_blocks = pair_diff_blocks(mySources, col=select_idx+'_blocks')
                block_labels = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
                block_labels_idx = ['select_idx_diff_'+a for a in block_labels]
                for i,block in enumerate(block_labels):
                    pair_types[block_labels_idx[i]] = (select_idx_diff_blocks[block])/2
                # generate pair metrics for individual blocks
                block_labels = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
                block_pair_types = dict()
                block_pair_types_detail = dict()

                for block in block_labels:
                    (block_pair_types[block],block_pair_types_detail[block]) =  calc_RL_pair_types(mySources, selectivity_type=selectivity+'_'+block)            
                    pair_types['pair_selectivity_'+block] = block_pair_types[block]
                    pair_types['pair_selectivity_'+block+'_detail'] = block_pair_types_detail[block]
            #pair_types['corr_trial_avg'] = myCorr_trial_avg 

            # calc metrics for shuffles
            if n_shuf>0:
                for shuf_idx in range(n_shuf):
                    # calc selectivity categories
                    #(source_selectivity, target_selectivity) = source_target_labels(mySources, field=shuf_metric+'_'+str(i))
                    #pair_types['source_selectivity'] = source_selectivity
                    #pair_types['target_selectivity'] = target_selectivity
                    #pair_types['RL_diff_RL'] = myDiff_RL

                    (source_selectivity, target_selectivity) = source_target_labels(mySources, field=shuf_metric+'_'+str(shuf_idx))
                    pair_types['source_selectivity_'+str(shuf_idx)] = source_selectivity
                    pair_types['target_selectivity_'+str(shuf_idx)] = target_selectivity
                    #pair_types['RL_diff'] = select_idx_diff
                    #pair_types['RL_sim'] = (2 - myDiff_ROC)/2
                    # generate pair metrics
                    #(my_RL_pair_types, my_RL_pair_types_detail) = calc_RL_pair_types(mySources,  selectivity_type='RL_selectivity')
                    #pair_types['pair_selectivity'] = my_RL_pair_types
                    #pair_types['pair_selectivity_detail'] = my_RL_pair_types_detail
                    (pair_selectivity, pair_selectivity_detail) = calc_RL_pair_types(mySources, selectivity_type='selectivity_'+str(shuf_idx))
                    pair_types['pair_selectivity_'+str(shuf_idx)] = pair_selectivity
                    pair_types['pair_selectivity_detail_'+str(shuf_idx)] = pair_selectivity_detail
                    
                    # generate pair metrics for individual blocks
                    block_labels = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
                    block_labels_shuf = ['pair_selectivity_'+a+'_'+str(shuf_idx) for a in block_labels]
                    block_pair_types = dict()
                    block_pair_types_detail = dict()

                    for block in block_labels_shuf:
                        (block_pair_types[block],block_pair_types_detail[block]) =  calc_RL_pair_types(mySources, selectivity_type=block)            
                        pair_types['pair_selectivity_'+block] = block_pair_types[block]
                        pair_types['pair_selectivity_'+block+'_detail'] = block_pair_types_detail[block]
                    #pair_types['corr_trial_avg'] = myCorr_trial_avg 
            return pair_types
        pair_types = calc_pair_metrics_shuf(mySources, myCN_df, 
            n_shuf = n_shuf, select_idx=select_idx, selectivity=selectivity)
        return pair_types
    pair_types= calc_conv_pairs(neurons_DF, cn_DF, select_idx=select_idx, selectivity=selectivity, n_shuf=n_shuf)
    pair_types = pd.DataFrame.from_dict(pair_types)
    #pair_types = pair_types[~np.isnan(pair_types.avg_conv_psd_area)]
    pair_types['source_select'] = pair_types.apply (lambda row: add_source_select(row, neurons_DF,select_idx=select_idx), axis=1)
    pair_types['target_select'] = pair_types.apply (lambda row: add_target_select(row, neurons_DF,select_idx=select_idx), axis=1)
    #pair_types['source_type'] = pair_types.apply (lambda row: add_source_type(row, neurons_DF), axis=1)
    pair_types['conv_target_type'] = pair_types.apply (lambda row: add_target_type(row, psp_DF), axis=1)
 
    return pair_types
     #conv_pair_df = conv_pair_df[~np.isnan(conv_pair_df['avg_conv_psd_area'])]
    #conv_pair_df['shuffle'] = 'data'
    #return conv_pair_df

def gen_block_conv_shuf(conv_DF, n_shuf=0, ymetric='avg_conv_psd_area',useMedian=False):
    colors = palette ={"Same": "green", "Different": "purple"}
    block_labels = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
    block_sel_labels = ['pair_selectivity_'+ i for i in block_labels]
    block_conv_df = pd.DataFrame([],columns=['psd_frac','block','shuffle','shuf_idx'])
    # data

    for (i,block) in enumerate(block_sel_labels):    
        myBlockConv = pd.Series()    
        if useMedian:
            myBlockConv['psd_frac'] = np.nanmedian(conv_DF[conv_DF[block]=='Same'][ymetric]) / np.nanmedian(conv_DF[conv_DF[block]=='Different'][ymetric])
            print(block + ' Same / Diff = Frac %f / %f / %f' % (np.nanmedian(conv_DF[conv_DF[block]=='Same'][ymetric]),np.nanmedian(conv_DF[conv_DF[block]=='Different'][ymetric]), myBlockConv['psd_frac']))
        else:
            myBlockConv['psd_frac'] = np.nanmean(conv_DF[conv_DF[block]=='Same'][ymetric]) / np.nanmean(conv_DF[conv_DF[block]=='Different'][ymetric])
            print(block + ' Same / Diff = Frac %f / %f / %f' % (np.nanmean(conv_DF[conv_DF[block]=='Same'][ymetric]),np.nanmean(conv_DF[conv_DF[block]=='Different'][ymetric]), myBlockConv['psd_frac']))
        myBlockConv['block'] = block_labels[i]
        myBlockConv['shuffle'] = 'data'
        myBlockConv['shuf_idx'] = np.nan
        block_conv_df = block_conv_df.append(myBlockConv, ignore_index=True)
    # shuf
    for shuf_idx in range(n_shuf):
        block_labels_shuf = [a+'_shuffle_'+str(shuf_idx) for a in block_sel_labels]
        for (i,block) in enumerate(block_labels_shuf):      
            myBlockConv = pd.Series(name=block)   
            if useMedian:
                myBlockConv['psd_frac'] = np.nanmedian(conv_DF[conv_DF[block]=='Same'][ymetric]) / np.nanmedian(conv_DF[conv_DF[block]=='Different'][ymetric])
            else:
                myBlockConv['psd_frac'] = np.nanmean(conv_DF[conv_DF[block]=='Same'][ymetric]) / np.nanmean(conv_DF[conv_DF[block]=='Different'][ymetric])
            myBlockConv['block'] = block_labels[i]
            myBlockConv['shuffle'] = 'shuffle'
            myBlockConv['shuf_idx'] = shuf_idx
            block_conv_df = block_conv_df.append(myBlockConv, ignore_index=False)
    return block_conv_df

def gen_block_neuron_DF_shuf(neuronDF, n_shuf):
    #colors = palette ={"Same": "green", "Different": "purple"}
    #block_labels = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
    block_labels  = ['_cueEarly','_cueLate','_delayEarly','_delayTurn','_turnITI']
    block_labels  = ['selectivity_ROC' + i for i in block_labels ]
    block_df = pd.DataFrame([],columns=['psd_frac','block','shuffle','shuf_idx'])
    # data

    for (i,block) in enumerate(block_labels):    
        myBlockSeries = pd.Series()    

        left_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Left'])].avg_out_psd_area)
        right_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Right'])].avg_out_psd_area)
        select_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Left','Right'])].avg_out_psd_area)
        non_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Non'])].avg_out_psd_area)
        myBlockSeries['left_psd_frac'] = left_psd / non_psd
        myBlockSeries['right_psd_frac'] = right_psd / non_psd
        myBlockSeries['select_psd_frac'] = select_psd / non_psd

        left_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Left'])].syn_density_collat)
        right_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Right'])].syn_density_collat)
        select_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Left','Right'])].syn_density_collat)
        non_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Non'])].syn_density_collat)
        myBlockSeries['left_collat_frac'] = left_collat / non_collat
        myBlockSeries['right_collat_frac'] = right_collat / non_collat
        myBlockSeries['select_collat_frac'] = select_collat / non_collat

        myBlockSeries['block'] = block
        myBlockSeries['shuffle'] = 'data'
        myBlockSeries['shuf_idx'] = np.nan
        block_df = block_df.append(myBlockSeries, ignore_index=True)
    # shuf
    for shuf_idx in range(n_shuf):
        block_labels_shuf = [a+'_shuffle_'+str(shuf_idx) for a in block_labels]
        for (i,block) in enumerate(block_labels_shuf):      
            myBlockSeries = pd.Series()    

            left_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Left'])].avg_out_psd_area)
            right_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Right'])].avg_out_psd_area)
            select_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Left','Right'])].avg_out_psd_area)
            non_psd = np.nanmedian(neuronDF[neuronDF[block].isin(['Non'])].avg_out_psd_area)
            myBlockSeries['left_psd_frac'] = left_psd / non_psd
            myBlockSeries['right_psd_frac'] = right_psd / non_psd
            myBlockSeries['select_psd_frac'] = select_psd / non_psd

            left_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Left'])].syn_density_collat)
            right_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Right'])].syn_density_collat)
            select_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Left','Right'])].syn_density_collat)
            non_collat = np.nanmedian(neuronDF[neuronDF[block].isin(['Non'])].syn_density_collat)
            myBlockSeries['left_collat_frac'] = left_collat / non_collat
            myBlockSeries['right_collat_frac'] = right_collat / non_collat
            myBlockSeries['select_collat_frac'] = select_collat / non_collat
            
            myBlockSeries['block'] = block_labels[i]
            myBlockSeries['shuffle'] = 'shuffle'
            myBlockSeries['shuf_idx'] = shuf_idx
            block_df = block_df.append(myBlockSeries, ignore_index=True)
    return block_df

def gen_conv_triads_DF(cn_DF, neurons_DF, actData, select_idx='select_idx_ROC', selectivity='selectivity_ROC', 
    dataset='PPC',n_shuf=0, blocks=None):
    print('%i seed neurons' % len(neurons_DF))
    print('%i total connections' % len(cn_DF))
    psp_skels = cn_DF.target.values
    (uniqueValues , indicesList, occurCount)= np.unique(psp_skels , return_index=True, return_counts=True)
    print('%i PSPs' % len(uniqueValues))
    print('%i PSPs receiving convergent input' % len(uniqueValues[occurCount>1]))

    conv_psp_skids = psp_skels[indicesList[occurCount>1]]
    conv_cn_DF = cn_DF[cn_DF.target.isin(conv_psp_skids)]
    conv_cn_DF.sort_values('target')

    conv_triads_DF = pd.DataFrame()
    # Loop over unique PSPs receiving conv input
    for target_skid in uniqueValues:
        #target_skid = conv_psp_skids[i]
        my_conv_cns = conv_cn_DF[conv_cn_DF.target==target_skid ]
        for source_skid_pair in combinations(my_conv_cns.source,2):
            (skid_A, skid_B) = source_skid_pair
            avg_psd_area_A = my_conv_cns[my_conv_cns.source==skid_A].avg_psd_area.values[0]
            avg_psd_area_B = my_conv_cns[my_conv_cns.source==skid_B].avg_psd_area.values[0]
            if avg_psd_area_A < avg_psd_area_B: # by convention, A is larger than B
                source_skid_pair = reversed(source_skid_pair)
            conv_triad = conv_triad_from_cns(my_conv_cns, source_skid_pair, target_skid)                       
            # generate pair metrics for individual blocks
            if blocks is not None:
                if dataset=='PPC' or dataset=='PPC_test':
                    block_labels = ['cueEarly','cueLate','delayEarly','delayTurn','turnITI']
                    for block in block_labels:
                        selA = my_conv_cns[my_conv_cns.source==skid_A]['source_selectivity_'+block].values[0]
                        selB = my_conv_cns[my_conv_cns.source==skid_B]['source_selectivity_'+block].values[0]
                        conv_triad['pair_selectivity_'+block] = calc_pair_selectivity(selA, selB)
            conv_triads_DF = conv_triads_DF.append(conv_triad, ignore_index=False)
    if dataset=='PPC' or dataset=='PPC_test':
        # add functional correlation metrics
        mySessions = actData['mySessions']
        conv_triads_DF['corr_trial_avg'] = conv_triads_DF.apply(lambda row: cAPI.act_corr(row['source_skid_A'], 
            row['source_skid_B'], actData['corr_trial_avg'], mySessions, neurons_DF), axis=1)
        conv_triads_DF['corr_raw'] = conv_triads_DF.apply (lambda row: cAPI.act_corr(row['source_skid_A'], 
            row['source_skid_B'], actData['corr_raw'], mySessions, neurons_DF), axis=1)
        conv_triads_DF['corr_residual'] = conv_triads_DF.apply (lambda row: cAPI.act_corr(row['source_skid_A'], 
            row['source_skid_B'], actData['corr_residual'], mySessions, neurons_DF), axis=1)
    if n_shuf > 0:
        for i in range(n_shuf):
            print('shuffle %i' % i)
            si = str(i)
            conv_triads_DF['selectivity_A_shuffle_'+si] = conv_triads_DF.apply(lambda row: lookup_val(neurons_DF,'skeleton_id',row['source_skid_A'],selectivity+'_shuffle_'+si),axis=1)
            conv_triads_DF['selectivity_B_shuffle_'+si] = conv_triads_DF.apply(lambda row: lookup_val(neurons_DF,'skeleton_id',row['source_skid_B'],selectivity+'_shuffle_'+si),axis=1)
            conv_triads_DF['pair_selectivity_shuffle_'+si] = conv_triads_DF.apply(lambda row: calc_pair_selectivity_opp(row['selectivity_A_shuffle_'+si], row['selectivity_B_shuffle_'+si]), axis=1)
            # blockwise selectivity
            '''if dataset == 'PPC' or dataset == 'PPC_test':
                for block in block_labels:
                    conv_triads_DF['selectivity_A_'+block+'_shuffle_'+si] = conv_triads_DF.apply(lambda row: lookup_val(neurons_DF,'skeleton_id',row['source_skid_A'],selectivity+'_'+block+'_shuffle_'+si),axis=1)
                    conv_triads_DF['selectivity_B_'+block+'_shuffle_'+si] = conv_triads_DF.apply(lambda row: lookup_val(neurons_DF,'skeleton_id',row['source_skid_B'],selectivity+'_'+block+'_shuffle_'+si),axis=1)
                    conv_triads_DF['pair_selectivity_'+block+'_shuffle_'+si] = conv_triads_DF.apply(lambda row: calc_pair_selectivity(row['selectivity_A_'+block+'_shuffle_'+si], row['selectivity_B_'+block+'_shuffle_'+si]), axis=1)
            '''
    # Calc number of triad types
    print('%i convergent triads, %i EE-to-E, %i EI-to-E, %i II-to-E, %i EE-to-I, %i EI-to-I, %i II-to-I' % 
        (len(conv_triads_DF),sum(conv_triads_DF.triad_type == 'EE-E'),sum(conv_triads_DF.triad_type == 'EI-E'),
        sum(conv_triads_DF.triad_type == 'II-E'),sum(conv_triads_DF.triad_type == 'EE-I'),
        sum(conv_triads_DF.triad_type == 'EI-I'),sum(conv_triads_DF.triad_type == 'II-I')))
    return conv_triads_DF

def conv_triad_from_cns(my_conv_cns, source_skid_pair, target_skid, select_idx = 'MI'):
    (skid_A, skid_B) = source_skid_pair
    conv_triad = pd.Series(name=tuple([source_skid_pair, target_skid]))
    conv_triad['source_skid_A'] = skid_A
    conv_triad['source_skid_B'] = skid_B
    conv_triad['target'] = target_skid

    conv_triad['source_type_A'] = my_conv_cns[my_conv_cns.source==skid_A].source_type.values[0]
    conv_triad['source_type_B'] = my_conv_cns[my_conv_cns.source==skid_B].source_type.values[0]
    conv_triad['target_type'] = my_conv_cns.target_type.values[0]
    conv_triad['triad_type'] = calc_triad_type(conv_triad['source_type_A'],conv_triad['source_type_B'],conv_triad['target_type'])

    conv_triad['cn_ids_A'] = my_conv_cns[my_conv_cns.source==skid_A].cn_ids.values[0]
    conv_triad['cn_ids_B'] = my_conv_cns[my_conv_cns.source==skid_B].cn_ids.values[0]
    conv_triad['avg_psd_area_A'] = my_conv_cns[my_conv_cns.source==skid_A].avg_psd_area.values[0]
    conv_triad['avg_psd_area_B'] = my_conv_cns[my_conv_cns.source==skid_B].avg_psd_area.values[0]
    conv_triad['log_avg_psd_area_A'] = np.log10(my_conv_cns[my_conv_cns.source==skid_A].avg_psd_area.values[0])
    conv_triad['log_avg_psd_area_B'] = np.log10(my_conv_cns[my_conv_cns.source==skid_B].avg_psd_area.values[0])
   
    conv_triad['avg_conv_psd_area'] = np.average([conv_triad['avg_psd_area_A'],conv_triad['avg_psd_area_B']])
    conv_triad['syn_count_A'] = my_conv_cns[my_conv_cns.source==skid_A].syn_count.values[0]
    conv_triad['syn_count_B'] = my_conv_cns[my_conv_cns.source==skid_B].syn_count.values[0]
    conv_triad['min_syn_count'] = np.minimum(conv_triad['syn_count_A'],conv_triad['syn_count_B'])
    conv_triad['avg_syn_count'] = np.average([conv_triad['syn_count_A'],conv_triad['syn_count_B']])
    conv_triad['select_idx_A']= my_conv_cns[my_conv_cns.source==skid_A].source_select_idx.values[0]
    conv_triad['select_idx_B']= my_conv_cns[my_conv_cns.source==skid_B].source_select_idx.values[0]
    conv_triad['cable_overlap_A']= my_conv_cns[my_conv_cns.source==skid_A].cable_overlap.values[0]
    conv_triad['cable_overlap_B']= my_conv_cns[my_conv_cns.source==skid_B].cable_overlap.values[0]

    if select_idx == 'oripeaksel':
        select_diff = np.abs(conv_triad['select_idx_A']-conv_triad['select_idx_B'])
        select_diff = np.minimum(select_diff, 180-select_diff)
        conv_triad['pair_selectivity'] = calc_ori_pair_sel(select_diff)
    elif select_idx == 'dirpeaksel':
        select_diff = np.abs(conv_triad['select_idx_A']-conv_triad['select_idx_B'])
        select_diff= np.minimum(select_diff, 360-select_diff)
    else: #PPC
        conv_triad['pair_select_idx'] = add_pair_select_idx(conv_triad['select_idx_A'],conv_triad['select_idx_B'])
        conv_triad['selectivity_A']= my_conv_cns[my_conv_cns.source==skid_A].source_selectivity.values[0]
        conv_triad['selectivity_B']= my_conv_cns[my_conv_cns.source==skid_B].source_selectivity.values[0]
        conv_triad['pair_selectivity']=calc_pair_selectivity_opp(conv_triad['selectivity_A'],conv_triad['selectivity_B'])
        conv_triad['pair_selectivity_detailed']=calc_pair_selectivity_detailed(conv_triad['selectivity_A'],conv_triad['selectivity_B'])
        conv_triad['pair_selectivity_all']=conv_triad['selectivity_A']+'_'+conv_triad['selectivity_B']
        conv_triad['tCOM_A'] = my_conv_cns[my_conv_cns.source==skid_A].source_tCOM.values[0]
        conv_triad['tCOM_B'] = my_conv_cns[my_conv_cns.source==skid_B].source_tCOM.values[0]
        conv_triad['tCOM_diff'] = conv_triad['tCOM_A']-conv_triad['tCOM_B']
        conv_triad['t_peakMI_A'] = my_conv_cns[my_conv_cns.source==skid_A].source_t_peakMI.values[0]
        conv_triad['t_peakMI_B'] = my_conv_cns[my_conv_cns.source==skid_B].source_t_peakMI.values[0]
        conv_triad['t_peakMI_diff'] = conv_triad['t_peakMI_A']-conv_triad['t_peakMI_B']
    return conv_triad
            
def gen_EIE_triads_DF(pyr_dir_cn_DF, nonpyr_dir_cn_DF):
    myE2I_cn_DF = pyr_dir_cn_DF[pyr_dir_cn_DF.target.isin(nonpyr_dir_cn_DF.source.values)]
   # print('%i E2X connections ' % len(pyr_dir_cn_DF))
    print('%i E2I connections ' % len(myE2I_cn_DF))
    myI2E_cn_DF = nonpyr_dir_cn_DF[nonpyr_dir_cn_DF.source.isin(myE2I_cn_DF.target.values)]
    #print('%i I2X connections ' % len(nonpyr_dir_cn_DF))
    print('%i I2E connections ' % len(myI2E_cn_DF))
    EIE_cn_DF = pd.DataFrame([],dtype=int)
    #    for IN_skid in myNonPyr_cn_DF.source.values:
    #        sources = myPyr_cn_DF.source.values
    #        targets = myNonPyr_cn_DF.target.values
    #        for source in sources:
    #            for target in targets:
    for (E2I_idx, E2I_cn) in myE2I_cn_DF.iterrows():
        for (I2E_idx, I2E_cn) in myI2E_cn_DF[myI2E_cn_DF.source==E2I_cn.target].iterrows():
            myEIE_pair = pd.Series(dtype=np.int)
            myEIE_pair['E_source'] = E2I_cn.source
            myEIE_pair['I_inter'] = E2I_cn.target
            myEIE_pair['E_target'] = I2E_cn.target
            myEIE_pair['source_selectivity'] = E2I_cn['source_selectivity'] #myPyr_cn_DF[np.logical_and(myPyr_cn_DF.source == source,myPyr_cn_DF.target == IN_skid)]['source_selectivity'].values[0]
            myEIE_pair['source_select_idx'] = E2I_cn['source_select_idx']#myPyr_cn_DF[np.logical_and(myPyr_cn_DF.source == source,myPyr_cn_DF.target == IN_skid)]['source_select'].values[0]
            myEIE_pair['EI_count'] = E2I_cn['syn_count']#myPyr_cn_DF[np.logical_and(myPyr_cn_DF.source == source,myPyr_cn_DF.target == IN_skid)]['syn_count'].values[0]
            myEIE_pair['EI_syn_den'] = E2I_cn['syn_den']
            myEIE_pair['EI_avg_psd_area'] = E2I_cn['avg_psd_area']#myPyr_cn_DF[np.logical_and(myPyr_cn_DF.source == source,myPyr_cn_DF.target == IN_skid)]['avg_psd_area'].values[0]
            myEIE_pair['EI_cn_strength'] = E2I_cn['cn_strength']#myPyr_cn_DF[np.logical_and(myPyr_cn_DF.source == source,myPyr_cn_DF.target == IN_skid)]['cn_strength'].values[0]

            myEIE_pair['IE_count'] = I2E_cn['syn_count']#myNonPyr_cn_DF[np.logical_and(myNonPyr_cn_DF.source == IN_skid,myNonPyr_cn_DF.target == target)]['syn_count'].values[0]
            myEIE_pair['IE_syn_den'] = I2E_cn['syn_den']#myNonPyr_cn_DF[np.logical_and(myNonPyr_cn_DF.source == IN_skid,myNonPyr_cn_DF.target == target)]['avg_psd_area'].values[0]
            myEIE_pair['IE_avg_psd_area'] = I2E_cn['avg_psd_area']#myNonPyr_cn_DF[np.logical_and(myNonPyr_cn_DF.source == IN_skid,myNonPyr_cn_DF.target == target)]['avg_psd_area'].values[0]
            myEIE_pair['IE_cn_strength'] = I2E_cn['cn_strength']#myNonPyr_cn_DF[np.logical_and(myNonPyr_cn_DF.source == IN_skid,myNonPyr_cn_DF.target == target)]['cn_strength'].values[0]
            myEIE_pair['target_selectivity'] = I2E_cn['target_selectivity']#myNonPyr_cn_DF[np.logical_and(myNonPyr_cn_DF.source == IN_skid,myNonPyr_cn_DF.target == target)]['target_selectivity'].values[0]
            myEIE_pair['target_select_idx'] = I2E_cn['target_select_idx']#myNonPyr_cn_DF[np.logical_and(myNonPyr_cn_DF.source == IN_skid,myNonPyr_cn_DF.target == target)]['target_select'].values[0]

            myEIE_pair['EIE_count_avg'] = np.average([myEIE_pair['EI_count'],myEIE_pair['IE_count']])
            myEIE_pair['EIE_psd_avg'] = np.average([myEIE_pair['EI_avg_psd_area'],myEIE_pair['IE_avg_psd_area']])           
            myEIE_pair['EIE_cn_strength_mult'] = myEIE_pair['EI_cn_strength']*myEIE_pair['IE_cn_strength']    
            myEIE_pair['EIE_syn_den_mult'] = myEIE_pair['EI_syn_den']*myEIE_pair['IE_syn_den']  
            myEIE_pair['EIE_cn_strength_avg'] = np.average([myEIE_pair['EI_cn_strength'],myEIE_pair['IE_cn_strength']])    
            myEIE_pair['EIE_syn_den_avg'] = np.average([myEIE_pair['EI_syn_den'],myEIE_pair['IE_syn_den']])    
            myEIE_pair['target_type'] = 'pyramidal'

            myEIE_pair['pair_select_idx'] = add_pair_select_idx(myEIE_pair['source_select_idx'],myEIE_pair['target_select_idx'])
            myEIE_pair['pair_selectivity'] = calc_pair_selectivity(myEIE_pair['source_selectivity'],myEIE_pair['target_selectivity'])
            EIE_cn_DF = EIE_cn_DF.append(myEIE_pair, ignore_index=True)
    return EIE_cn_DF

def gen_pot_cn_DF(sources, targets, synapse_DF, actData = None, select_idx = 'select_idx_ROC', selectivity = 'selectivity_ROC',
        min_axon_len=0, min_synout=0, add_select=True, dir_cns = True, cable_overlaps=True, min_overlap = None, dataset='PPC',
        dendrite_type = 'dendrite'):
    mySources = sources
    mySources = mySources[mySources.axon_len_collats>min_axon_len] 
    mySources = mySources[mySources.num_axon_synapses>min_synout]
    myTargets = targets
    print('Source neurons: %i' % len(mySources))
    print('Target neurons: %i' % len(myTargets))
    edg_lst = cAPI.gen_potential_cns_df(mySources,myTargets)
    edg_lst.rename(columns={"weight": "syn_count"}, inplace = True)
    edg_lst['psd_areas'] = edg_lst.apply (lambda row: add_psd_areas(row, synapse_DF), axis=1)
    edg_lst['avg_psd_area'] = edg_lst.apply (lambda row: add_avg_psd_area(row, synapse_DF), axis=1)
    edg_lst['cn_ids'] = edg_lst.apply (lambda row: add_connector_ids(row, synapse_DF), axis=1)
    if add_select:
        if dataset == 'PPC' or dataset == 'PPC_test':
            add_act_corrs = True
        else:
            add_act_corrs = False
        edg_lst = add_select_to_cnDF(edg_lst, mySources, myTargets, actData, synapse_DF = synapse_DF, 
            dir_cns = dir_cns, select_idx = select_idx, selectivity=selectivity, dataset=dataset,add_act_corrs = add_act_corrs)
    if cable_overlaps:
        print('adding cable overlaps')
        edg_lst = add_cable_overlaps(edg_lst, max_dist = '5 microns', dendrite_type=dendrite_type, proximal_r=my_prx_r)
        if min_overlap is not None:
            edg_lst = edg_lst[edg_lst.cable_overlap>min_overlap]
    return edg_lst

def add_select_to_cnDF(cn_DF, neurons_DF, psp_DF, actData=None, synapse_DF=None, dir_cns = True, cable_overlaps=False, 
    select_idx = 'select_idx_MI',selectivity = 'selectivity_MI',add_act_corrs=False, dendrite_type = 'dendrite', dataset='PPC'):
    # Add more fields  
    cn_DF['source_type'] = cn_DF.apply (lambda row: add_source_type(row, neurons_DF), axis=1)
    cn_DF['target_type'] = cn_DF.apply (lambda row: add_target_type(row, psp_DF), axis=1)
    cn_DF['cn_type'] = cn_DF.apply (lambda row: add_cn_type(row['source_type'], row['target_type']), axis =1)
    cn_DF['source_select_idx'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx=select_idx), axis=1)
    cn_DF['source_select_idx_new'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='select_idx_MI_new'), axis=1)
    
    cn_DF['source_select_idx_t'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='select_idx_MI_t'), axis=1)
    cn_DF['source_select_idx_t_new'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='select_idx_MI_t_new'), axis=1)
    
    #cn_DF['source_select_idx_t_filt'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='select_idx_MI_t_filt'), axis=1)
    #cn_DF['source_select_idx_t_filt_95'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='select_idx_MI_t_filt_95'), axis=1)
    #cn_DF['source_select_idx_t_filt_68'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='select_idx_MI_t_filt_68'), axis=1)
  
    #cn_DF['source_sessions_select_idx_t'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='sessions_select_idx_MI_t'), axis=1)
    #cn_DF['source_select_idx_std'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF, select_idx=select_idx+'_std'), axis=1)
    #cn_DF['source_select_idx_stdmean'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF, select_idx=select_idx+'_stdmean'), axis=1)
    #cn_DF['source_selectivity'] = cn_DF.apply (lambda row: add_source_select_class(row, neurons_DF, selectivity=selectivity), axis=1)
    if dir_cns:
        cn_DF['target_select_idx'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx=select_idx), axis=1)
        cn_DF['target_select_idx_new'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='select_idx_MI_new'), axis=1)
        
        cn_DF['target_select_idx_t'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='select_idx_MI_t'), axis=1)
        cn_DF['target_select_idx_t_new'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='select_idx_MI_t_new'), axis=1)
        
        #cn_DF['target_select_idx_t_filt'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='select_idx_MI_t_filt'), axis=1)
        #cn_DF['target_select_idx_t_filt_95'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='select_idx_MI_t_filt_95'), axis=1)
        #cn_DF['target_select_idx_t_filt_68'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='select_idx_MI_t_filt_68'), axis=1)
        #cn_DF['target_sessions_select_idx_t'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='sessions_select_idx_MI_t'), axis=1)
        #cn_DF['target_selectivity'] = cn_DF.apply (lambda row: add_target_select_class(row, psp_DF, selectivity=selectivity), axis=1)
        if dataset == 'PPC' or dataset == 'PPC_test':
            #cn_DF['source_tCOM'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='tCOM'), axis=1)
            #cn_DF['target_tCOM'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='tCOM'), axis=1)
            #cn_DF['tCOM_diff'] = cn_DF['target_tCOM']-cn_DF['source_tCOM']
            #cn_DF['source_t_peakMI'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF,select_idx='choiceMI_max_idx'), axis=1)
            #cn_DF['target_t_peakMI'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF,select_idx='choiceMI_max_idx'), axis=1)
            #cn_DF['t_peak_diff'] = cn_DF['target_t_peakMI']-cn_DF['source_t_peakMI']
            cn_DF['source_epoch'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF, select_idx='choiceMI_max_idx_epoch'), axis=1)
            cn_DF['target_epoch'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF, select_idx='choiceMI_max_idx_epoch'), axis=1)
            cn_DF['pair_epoch'] = cn_DF.apply (lambda row: compare_epochs(row['source_epoch'], row['target_epoch']), axis=1)
            #cn_DF['target_select_idx_std'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF, select_idx=select_idx+'_std'), axis=1)
            #cn_DF['target_select_idx_stdmean'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF, select_idx=select_idx+'_stdmean'), axis=1)
            cn_DF['pair_select_idx'] =  cn_DF.apply (lambda row: add_pair_select_idx(row['source_select_idx'], row['target_select_idx']), axis=1)
            cn_DF['pair_select_idx_new'] =  cn_DF.apply (lambda row: add_pair_select_idx(row['source_select_idx_new'], row['target_select_idx_new']), axis=1)
            
            cn_DF['pair_select_idx_t'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t'], row['target_select_idx_t'], type='t'), axis=1)
            cn_DF['pair_select_idx_t_new'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_new'], row['target_select_idx_t_new'], type='t'), axis=1)
            
            #cn_DF['pair_select_idx_t_filt'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_filt'], row['target_select_idx_t_filt'], type='t'), axis=1)
            cn_DF['pair_select_idx_tmax'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t'], row['target_select_idx_t'], type='max'), axis=1)
            cn_DF['pair_select_idx_tmax_new'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_new'], row['target_select_idx_t_new'], type='max'), axis=1)
            
            cn_DF['pair_select_idx_tavg'] = cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t'], row['target_select_idx_t'], type = 'avg'), axis=1)
            cn_DF['pair_select_idx_tavg_new'] = cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_new'], row['target_select_idx_t_new'], type = 'avg'), axis=1)

            #cn_DF['pair_select_idx_tmax_filt'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_filt'], row['target_select_idx_t_filt'], type='max'), axis=1)
            #cn_DF['pair_select_idx_tmax_filt_95'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_filt_95'], row['target_select_idx_t_filt_95'], type='max'), axis=1)
            #cn_DF['pair_select_idx_tmax_filt_68'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_filt_68'], row['target_select_idx_t_filt_68'], type='max'), axis=1)
            #cn_DF['pair_select_idx_tavg_filt'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_filt'], row['target_select_idx_t_filt'], type = 'avg'), axis=1)
            #cn_DF['pair_select_idx_tavg_filt_95'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_filt_95'], row['target_select_idx_t_filt_95'], type = 'avg'), axis=1)
            #cn_DF['pair_select_idx_tavg_filt_68'] =  cn_DF.apply (lambda row: add_pair_select_idx_t(row['source_select_idx_t_filt_68'], row['target_select_idx_t_filt_68'], type = 'avg'), axis=1)
            #cn_DF['pair_select_idx_tmax_epochs'] =  cn_DF.apply (lambda row: add_pair_select_idx_tmax_epochs(row, neurons_DF, psp_DF), axis=1)
            #cn_DF['pair_select_idx_tmaxsess'] =  cn_DF.apply (lambda row: add_pair_select_idx_t_sess(row['source_sessions_select_idx_t'], 
            #    row['target_sessions_select_idx_t'], type='max'), axis=1)
            #cn_DF['pair_avg_select_idx'] =  cn_DF.apply (lambda row: add_avg_pair_select_idx(row['source_select_idx_t'], row['target_select_idx_t']), axis=1)
            
            #cn_DF['pair_selectivity'] = cn_DF.apply (lambda row: add_RL_pair_types_opp(row, neurons_DF), axis=1)
            #cn_DF['pair_select_idx_std'] = cn_DF.apply (lambda row: add_pair_select_idx_stdmean(row['source_select_idx'], row['target_select_idx'], 
            #    row['source_select_idx_std'], row['target_select_idx_std']), axis=1)
            #cn_DF['pair_select_idx_stdmean'] = cn_DF.apply (lambda row: add_pair_select_idx_stdmean(row['source_select_idx'], row['target_select_idx'], 
            #    row['source_select_idx_stdmean'], row['target_select_idx_stdmean']), axis=1)
            if add_act_corrs: # Add activity correlations
                cn_DF['corr_trial_avg'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['corr_trial_avg'], actData['mySessions'], psp_DF), axis=1)
                cn_DF['corr_raw'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['corr_raw'], actData['mySessions'], psp_DF), axis=1)
                cn_DF['corr_residual'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['corr_residual'], actData['mySessions'], psp_DF), axis=1)
                
                # Adding check and other corr measures 220623
                cn_DF['pearsonCorr_all'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_all'), axis=1)
                cn_DF['pearsonCorr_lr'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_Ca_lr'), axis=1)
                cn_DF['pearsonCorr_trialMean'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_Ca_trialMean'), axis=1)
                cn_DF['pearsonCorr_trialResidual'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_Ca_residual'), axis=1)
                
                # Adding more corr measures 220802
                cn_DF['pearsonCorr_trialMean_all'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_trialMean_all'), axis=1)
                cn_DF['pearsonCorr_trialResidual_bwavg'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_trialResidual_bwavg'), axis=1)
                cn_DF['pearsonCorr_bw_diff'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_bw_diff'), axis=1)
                cn_DF['pearsonCorr_timeMean'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_timeMean'), axis=1)
                
                # Adding more corr measures 230127
                cn_DF['pearsonCorr_lr_bin'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'corr_lr_bin'), axis=1)
                cn_DF['frac_coincident'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pearsonCorr'], actData['mySessions'], psp_DF, field = 'frac_coincident'), axis=1)
                # Adding pair_select_sync 230130
                cn_DF['pair_select_sync'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pair_select_sync'], actData['mySessions'], psp_DF, field = 'pair_select_idx'), axis=1)
                
                #cn_DF['pairMI_lr'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pairMI'], actData['mySessions'], psp_DF, field = 'lr'), axis=1)
                #cn_DF['pairMI_trialResidual'] = cn_DF.apply (lambda row: cAPI.act_corr(row['source'], row['target'], actData['pairMI'], actData['mySessions'], psp_DF, field = 'trialResidual'), axis=1)
#with open('local_data/2P_data_PPC_corrMetrics_210804.pkl','rb') as f:
#    (actData['pairMI'], actData['pearsonCorr']) = pickle.load(f)
#dict_keys(['corr_all', 'corr_Ca_lr', 'corr_Ca_trialMean', 'corr_Ca_residual'])
#dict_keys(['lr', 'trialResidual'])

        elif dataset == 'V1' or dataset == 'V1_test':
            cn_DF['source_oritunsel'] = cn_DF.apply (lambda row: add_source_select(row, neurons_DF, select_idx='oritunsel'), axis=1)
            cn_DF['target_oritunsel'] = cn_DF.apply (lambda row: add_target_select(row, psp_DF, select_idx='oritunsel'), axis=1)
            cn_DF['pair_select_idx'] =  cn_DF.apply (lambda row: add_ori_pair_idx(row, avg=None), axis=1)
            cn_DF['oripeakdiff'] =  cn_DF.apply (lambda row: add_ori_pair_idx_old(row), axis=1)
            cn_DF['pair_selectivity'] = cn_DF.apply (lambda row: add_ori_pair_types(row, neurons_DF), axis=1)
    #if synapse_DF is not None:
    #    cn_DF['psd_areas'] = cn_DF.apply (lambda row: add_psd_areas(row, synapse_DF), axis=1)
    #    cn_DF['avg_psd_area'] = cn_DF.apply (lambda row: add_avg_psd_area(row, synapse_DF), axis=1)
    #    cn_DF['cn_ids'] = cn_DF.apply (lambda row: add_connector_ids(row, synapse_DF), axis=1)
    if cable_overlaps:
        cn_DF = add_cable_overlaps(cn_DF, max_dist = '5 microns', dendrite_type=dendrite_type, proximal_r=my_prx_r)
        cn_DF['log_syn_den'] = np.log10(cn_DF.syn_den.values)
        cn_DF['syn_den_norm'] =  cn_DF.apply (lambda row: add_syn_den_norm(row, neurons_DF), axis=1)
        cn_DF['log_syn_den_norm'] = np.log10(cn_DF.syn_den_norm.values)
    return cn_DF

def get_cn_rate_DF(pot_dir_cn_DF, n_shuf = 0, soma_dist_bins = np.linspace(0,400,15), max_soma_dist=None,
    corr_bins = [-1,.1,1],min_overlap = None,cable_overlap_bins = np.linspace(0,200,15), datasets = 'PPC', count_mult=False):

    cn_rate_DF = pd.DataFrame()
    cn_rate_overlap_DF = pd.DataFrame()
    cn_rate_avg_DF = pd.DataFrame()
    cn_rate_avg_overlap_DF = pd.DataFrame()
    cn_rate_bins_DF = pd.DataFrame() 
    cn_rate_overlap_bins_DF = pd.DataFrame()
    cn_rate_bins_overlap_DF = pd.DataFrame() 
    cn_type_lbl = np.array([['E-to-E','E-to-I'],['I-to-E','I-to-I']])
 
    edg_lst = pot_dir_cn_DF
    for i,source_type in enumerate(['pyramidal','non pyramidal']):
        for j,target_type in enumerate(['pyramidal', 'non pyramidal']):
            edg_lst_sel = sel_edges(edg_lst, source_type=source_type, target_type=target_type, max_soma_dist=max_soma_dist)    
            #edg_lst_sel['corr_binned']=pd.cut(edg_lst_sel['corr_trial_avg'],corr_bins,labels=corr_bins[:-1])    
            #for corr_bin in corr_bins:
            #    edg_lst_bin = edg_lst_sel[edg_lst_sel.corr_binned == corr_bin]
            #    cn_rate_bin = cAPI.cn_rate_from_pot_cns(edg_lst_bin, cn_type=cn_type_lbl[i,j])
            #    cn_rate_bin['corr_bin']=corr_bin
            #    cn_rate_bins_DF = cn_rate_bins_DF.append(cn_rate_bin, ignore_index=True)
            for pair_selectivity in ['Same', 'Non', 'Opp']:                
                edg_lst_sel = sel_edges(edg_lst, source_type=source_type, target_type=target_type,
                pair_selectivity=pair_selectivity, max_soma_dist=max_soma_dist, min_overlap=min_overlap)
                cn_rate_avg = cAPI.cn_rate_from_pot_cns(edg_lst_sel, count_mult=count_mult,
                    cn_type=cn_type_lbl[i,j], label=pair_selectivity)
                cn_rate_avg['pair_selectivity']=pair_selectivity
                cn_rate_avg_DF = cn_rate_avg_DF.append(cn_rate_avg, ignore_index=True) 
                #cn_rate_avg_overlap = cAPI.cn_rate_from_pot_cns(edg_lst_sel[edg_lst_sel.cable_overlap>0], cn_type=cn_type_lbl[i,j], label=pair_selectivity)
                #cn_rate_avg_overlap['pair_selectivity']=pair_selectivity
                #cn_rate_avg_overlap_DF = cn_rate_avg_overlap_DF.append(cn_rate_avg_overlap, ignore_index=True) 
                if n_shuf==0:
                    cn_rate = cn_rate_avg
                    cn_rate['shuf'] = 'data'
                    #cn_rate_overlap = cn_rate_avg_overlap
                    cn_rate_DF = cn_rate_DF.append(cn_rate, ignore_index=True)
                else:
                    for shuf in range(n_shuf):
                        if shuf == 0:
                            cn_rate = cn_rate_avg
                            cn_rate['shuf'] = 'data'
                        else:
                            cn_rate = cAPI.cn_rate_from_pot_cns(edg_lst_sel.sample(frac=1, replace=True), count_mult=count_mult, 
                                cn_type=cn_type_lbl[i,j], label=pair_selectivity)
                            #cn_rate_overlap = cAPI.cn_rate_from_pot_cns(edg_lst_sel[edg_lst_sel.cable_overlap>0].sample(frac=1, replace=True), cn_type=cn_type_lbl[i,j], label=pair_selectivity)
                            cn_rate['pair_selectivity']=pair_selectivity
                            cn_rate['shuf'] = 'shuf'
                        cn_rate_DF = cn_rate_DF.append(cn_rate, ignore_index=True)
    return cn_rate_DF

def gen_pot_conv_triads_DF(MN_DF, psp_DF, syn_DF, pot_cn_DF, max_soma_dist = None):
    # only consider possible connections within max_soma_dist
    if max_soma_dist is not None:
        pot_cn_DF = pot_cn_DF[pot_cn_DF.soma_dist < max_soma_dist]
    # only consider connections among cells with identified cell type
    pot_cn_DF = pot_cn_DF[pot_cn_DF.target_type != 'unknown']

    targets = np.unique(pot_cn_DF.target)
    sources = np.unique(pot_cn_DF.source)

    pot_conv_triads_DF = pd.DataFrame()
    for i,target in enumerate(targets):
        myTarget_pot_cn_DF = pot_cn_DF[pot_cn_DF.target == target]
        mySources = myTarget_pot_cn_DF.source
        print('Eval target %i of %i with %i potential sources' % (i, len(targets), len(mySources)))
        if len(mySources) < 2:
            mySourcePairs = None
        else:
            mySourcePairs = itertools.combinations(mySources, 2)
        if mySourcePairs is not None:
            for source_skid_pair in mySourcePairs:
                (skid_A, skid_B) = source_skid_pair
                avg_psd_area_A = myTarget_pot_cn_DF[myTarget_pot_cn_DF.source==skid_A].avg_psd_area.values[0]
                avg_psd_area_B = myTarget_pot_cn_DF[myTarget_pot_cn_DF.source==skid_B].avg_psd_area.values[0]
                if avg_psd_area_A < avg_psd_area_B: # by convention, A is larger than B
                    source_skid_pair = reversed(source_skid_pair)
                conv_triad = conv_triad_from_cns(myTarget_pot_cn_DF, source_skid_pair, target)    
                pot_conv_triads_DF = pot_conv_triads_DF.append(conv_triad, ignore_index=False)
    return pot_conv_triads_DF

def get_conv_triads_rate_DF(pot_conv_triads_DF, n_shuf = 0, soma_dist_bins = np.linspace(0,400,15), syn_count='min_syn_count',
    corr_bins = [-1,.1,1],cable_overlap_bins = np.linspace(0,200,15), datasets = 'PPC', count_mult=False):

    cn_rate_DF = pd.DataFrame()
    cn_rate_overlap_DF = pd.DataFrame()
    cn_rate_avg_DF = pd.DataFrame()
    cn_rate_avg_overlap_DF = pd.DataFrame()
    cn_rate_bins_DF = pd.DataFrame() 
    cn_rate_overlap_bins_DF = pd.DataFrame()
    cn_rate_bins_overlap_DF = pd.DataFrame() 
    cn_type_lbl = np.array([['E-to-E','E-to-I'],['I-to-E','I-to-I']])

    edg_lst = pot_conv_triads_DF
    for triad_type in ('EE-E','EI-E','EE-I','EI-I','II-E','II-I'):
        edg_lst_type = sel_edges(edg_lst, triad_type=triad_type)    
        for pair_selectivity in ['Same', 'Non', 'Opp']:
            edg_lst_type_pair = sel_edges(edg_lst_type, pair_selectivity=pair_selectivity)
            cn_rate_avg = cAPI.cn_rate_from_pot_cns(edg_lst_type_pair, count_mult=count_mult,
                cn_type=triad_type, label=pair_selectivity, syn_count=syn_count)
            cn_rate_avg['pair_selectivity']=pair_selectivity
            cn_rate_avg_DF = cn_rate_avg_DF.append(cn_rate_avg, ignore_index=True) 
            #cn_rate_avg_overlap = cAPI.cn_rate_from_pot_cns(edg_lst_sel[edg_lst_sel.cable_overlap>0], cn_type=cn_type_lbl[i,j], label=pair_selectivity)
            #cn_rate_avg_overlap['pair_selectivity']=pair_selectivity
            #cn_rate_avg_overlap_DF = cn_rate_avg_overlap_DF.append(cn_rate_avg_overlap, ignore_index=True) 
            if n_shuf==0:
                cn_rate = cn_rate_avg
                cn_rate['shuf'] = 'data'
                #cn_rate_overlap = cn_rate_avg_overlap
                cn_rate_DF = cn_rate_DF.append(cn_rate, ignore_index=True)
            else:
                for shuf in range(n_shuf):
                    if shuf == 0:
                        cn_rate = cn_rate_avg
                        cn_rate['shuf'] = 'data'
                    else:
                        cn_rate = cAPI.cn_rate_from_pot_cns(edg_lst_type_pair.sample(frac=1, replace=True), count_mult=count_mult, 
                            cn_type=triad_type, label=pair_selectivity, syn_count='min_syn_count')
                        #cn_rate_overlap = cAPI.cn_rate_from_pot_cns(edg_lst_sel[edg_lst_sel.cable_overlap>0].sample(frac=1, replace=True), cn_type=cn_type_lbl[i,j], label=pair_selectivity)
                        cn_rate['pair_selectivity']=pair_selectivity
                        cn_rate['shuf'] = 'shuf'
                    cn_rate_DF = cn_rate_DF.append(cn_rate, ignore_index=True)
    return cn_rate_DF

def sel_edges(edg_lst, source_type=None, target_type=None, pair_selectivity=None, max_soma_dist=None, triad_type=None, min_overlap=None):
    edg_lst_sel = edg_lst
    if triad_type is not None:
        edg_lst_sel = edg_lst_sel[edg_lst_sel.triad_type==triad_type]
    if source_type is not None:
        edg_lst_sel = edg_lst_sel[edg_lst_sel.source_type==source_type]
    if target_type is not None:
        edg_lst_sel = edg_lst_sel[edg_lst_sel.target_type==target_type]
    if pair_selectivity is not None:
        edg_lst_sel = edg_lst_sel[edg_lst_sel.pair_selectivity==pair_selectivity]
    if max_soma_dist is not None:
        edg_lst_sel = edg_lst_sel[edg_lst_sel.soma_dist<max_soma_dist]
    if min_overlap is not None:
        edg_lst_sel = edg_lst_sel[edg_lst_sel.cable_overlap>min_overlap]
    return edg_lst_sel


def calc_local_cns(dataset, shuf_frac = None, min_collat_len = 100, min_syn=5):
    workingDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/'
    if dataset == 'PPC' or dataset=='PPC_test':
        rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                                api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff',
                                project_id=1)
        with open(workingDir+'MN_DF_'+dataset+'.pkl' , 'rb') as f:  
            mySources = pickle.load(f)
    elif dataset == 'V1' or dataset=='V1_test':
        rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                                    api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff',
                                    project_id = 31)
        # use "V1_sources" instead - has traced inh neurons not included in MN list
        with open('local_data/MN_neuronDF_V1_sources.pkl','rb') as f: 
            mySources = pickle.load(f)
    # Load potential targets 
    #with open('local_data/all_'+dataset+'.pkl' , 'rb') as f:  
    with open(workingDir+'typed_DF_'+dataset+'.pkl' , 'rb') as f:  
        myTargets = pickle.load(f)
    # Only include neurons with soma in vol
    mySources = mySources[mySources.has_soma == 1]
    myTargets = myTargets[myTargets.has_soma == 1]
    mySources = mySources[mySources.axon_len_collats>min_collat_len] # require 100 um axon collats
    mySources = mySources[mySources.num_axon_synapses>min_syn] # require at least 5 synapses (sanity check)

    if shuf_frac is not None:
        mySources = mySources.sample(frac = shuf_frac, replace = False)
        myTargets = mySources.sample(frac = shuf_frac, replace = False)
    edg_lst = cAPI.gen_potential_cns_df(mySources, myTargets)
    return edg_lst

def calc_local_cn_rates(max_soma_dist = 200, min_collat_len=100, min_syn=5):
    cn_rate_DF = pd.DataFrame()
    cn_rate_avg_DF = pd.DataFrame()
    bins = np.linspace(0,400,15)
    cn_type_lbl = np.array([['E-to-E','E-to-I'],['I-to-E','I-to-I']])
    datasets = ['V1','PPC']
    for dataset in datasets:
        edg_lst = calc_local_cns(dataset, min_collat_len=min_collat_len, min_syn=min_syn)
        for i,source_type in enumerate(['pyramidal','non pyramidal']):
            edg_lst_src = edg_lst[edg_lst.source_type==source_type]
            print('%s %s source neurons: %i' % (dataset,source_type,len(np.unique(edg_lst_src.source.values))))
            for j,target_type in enumerate(['pyramidal', 'non pyramidal']):
                edg_lst_src_tgt = edg_lst_src[edg_lst_src.target_type==target_type]
                edg_lst_trunc = edg_lst_src_tgt[edg_lst_src_tgt.soma_dist<max_soma_dist] # truncate max soma dist 
                cn_rate_avg = cAPI.cn_rate_from_pot_cns(edg_lst_trunc, cn_type=cn_type_lbl[i,j], label=dataset, syn_count = 'weight')
                cn_rate_avg_DF = cn_rate_avg_DF.append(cn_rate_avg, ignore_index=True) 
                for source_skid in np.unique(edg_lst_trunc.source.values):
                    cn_rate = cAPI.cn_rate_from_pot_cns(edg_lst_trunc[edg_lst_trunc.source==source_skid], cn_type=cn_type_lbl[i,j], 
                        label=dataset, syn_count = 'weight')
                    cn_rate['source_skid'] = source_skid
                    cn_rate_DF = cn_rate_DF.append(cn_rate, ignore_index=True) 
    return (cn_rate_DF, cn_rate_avg_DF)

## smaller misc functions
def lookup_val(ref_DF, ref_field, ref_val, lookup_field):
    try: 
        return ref_DF[ref_DF[ref_field]==ref_val][lookup_field].values[0]
    except:
        return np.nan
def add_source_loc(syn_row, neuron_df):
    try:
        return neuron_df[(neuron_df.skeleton_id==syn_row.source)]['soma_loc'].values[0]
    except:
        return np.nan
def add_source_type(row, ref_df):
    try:
        #return syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)].iloc[0]['source_type']
        return ref_df[ref_df.skeleton_id==row.source].iloc[0]['type']
    except:
        return None
def add_target_type(row, ref_df):
    try:
        #return syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)].iloc[0]['target_type']
        return ref_df[ref_df.skeleton_id==row.target].iloc[0]['type']
    except:
        return None
def add_cn_type(src_type, tgt_type):
    if src_type == 'pyramidal':
        if tgt_type == 'pyramidal':
            return 'E-E'
        elif tgt_type == 'non pyramidal':
            return 'E-I'
        else:
            return None
    elif src_type == 'non pyramidal':
        if tgt_type == 'pyramidal':
            return 'I-E'
        elif tgt_type == 'non pyramidal':
            return 'I-I'
        else:
            return None
def add_source_select(row, neuron_df, select_idx='select_idx_ROC'):
    try:
        return neuron_df[(neuron_df.skeleton_id==row.source)][select_idx].values[0]
    except:
        #print('skelID %i has no selectivity' % row.source)
        return np.nan

def add_syn_den_norm(row, neuron_df):
    try:
        return row['syn_den']/neuron_df[(neuron_df.skeleton_id==row.source)]['syn_density_collat'].values[0]
    except:
        return np.nan
def add_target_select(row, neuron_df, select_idx='select_idx_ROC'):
    try:
        return neuron_df[(neuron_df.skeleton_id==row.target)][select_idx].values[0]
    except:
        #print('skelID %i has no selectivity' % row.target)
        return np.nan
def add_source_select_class(row, neuron_df, selectivity='selectivity_ROC'):
    try:
        return neuron_df[(neuron_df.skeleton_id==row.source)][selectivity].values[0]
    except:
        #print('skelID %i has no selectivity class' % row.source)
        return np.nan
def add_target_select_class(row, neuron_df, selectivity='selectivity_ROC'):
    try:
        return neuron_df[(neuron_df.skeleton_id==row.target)][selectivity].values[0]
    except:
        #print('skelID %i has not selectivity class' % row.target)
        return np.nan
def add_pair_select_idx_OLD(xi, yi):
    mag = np.abs(xi+yi)/2
    diff = (2-np.abs(xi-yi))/2
    pair_selectivity = 1 - mag * diff
    return(pair_selectivity)

def add_pair_select_idx(xi, yi):
    if xi == 0 or yi == 0:
        mag = 0
    else:
        mag = scipy.stats.mstats.gmean([abs(xi), abs(yi)])
    # check if signs are the same
    if xi * yi > 0: # same
        pair_selectivity = mag
    else:
        pair_selectivity = -mag
    return(pair_selectivity)

def add_pair_select_idx_t(xt, yt, type = 't'):
    tpts = np.size(xt)
    pair_select_idx_t = np.empty(tpts)
    pair_select_idx_t[:] = np.nan
    mag = np.empty(tpts)
    mag[:] = np.nan
    if np.isnan(xt).any() or np.isnan(yt).any():
        return np.nan
    else:
        for t in np.arange(tpts):
            (xi, yi) = (xt[t], yt[t])
            if xi == 0 or yi == 0:
                mag[t] = 0
            else:
                if type == 'avg':
                    mag[t] = abs(xi) * abs(yi)
                    #mag[t] = scipy.stats.mstats.gmean([abs(xi), abs(yi)])
                else:
                    mag[t] = scipy.stats.mstats.gmean([abs(xi), abs(yi)])
            # check if signs are the same
            if xi * yi > 0: # same
                pair_select_idx_t[t] = mag[t]
            else:
                pair_select_idx_t[t] = -mag[t]     
        max_idx = np.argmax(mag)  
        if type == 't':
            return(pair_select_idx_t)
        elif type == 'max':
            return(pair_select_idx_t[max_idx])
        elif type == 'avg':
            mag_total = np.abs(np.mean(pair_select_idx_t))
            sign_total = np.sign(np.mean(pair_select_idx_t))
            return(sign_total * mag_total**(1/2))
            #return(np.mean(pair_select_idx_t))

def add_pair_select_idx_t_sess(sess_xt, sess_yt, type = 't'):
    num_sess = len(sess_xt)
    my_pair_select_idx = np.nan
    my_sum = 0
    num_active_sessions = 0
    sessions_select_idx = np.empty(num_sess)
    sessions_select_idx[:] = np.nan
    num_active_sessions = 0
    for i in range(num_sess):
        if np.size(sess_xt) > 1 and np.size(sess_yt) > 1: # in case they are just nan
            if not np.isnan(sess_xt[i]).any() and not np.isnan(sess_yt[i]).any():
                num_active_sessions += 1
                select_idx = add_pair_select_idx_t(sess_xt[i], sess_yt[i], type = type)
                sessions_select_idx[i] = select_idx
                my_sum += select_idx
    if num_active_sessions > 0:
        my_pair_select_idx = my_sum/num_active_sessions
    return my_pair_select_idx


def add_pair_select_idx_tmax_epochs(row, neurons_DF, psp_DF):
    epochs = ['cueEarly','cueLate','delay','turn','ITI']
    pair_select_idx_epochs = np.zeros(5)
    for (i,epoch) in enumerate(epochs):
        xi = add_source_select(row, neurons_DF,select_idx='select_idx_MI_'+epoch)
        yi = add_target_select(row, neurons_DF,select_idx='select_idx_MI_'+epoch)
        pair_select_idx_epochs[i] = add_pair_select_idx(xi, yi)
    max_i = np.argmax(np.abs(pair_select_idx_epochs))
    return pair_select_idx_epochs[max_i]

def add_pair_select_idx_stdmean(s1, s2, ds1, ds2):
    (s1, s2, ds1, ds2) = np.abs((s1, s2, ds1, ds2))
    return (ds1 * s2 + s1 * ds2) * (s1 * s2)**(-.5) / 2 
def calc_pair_selectivity_opp(RL_1, RL_2):
    try:
        if not RL_1 in ['Non','Right','Left','Mixed']:
            return np.nan
        elif not RL_2 in ['Non','Right','Left','Mixed']:
            return np.nan
        elif RL_1 == 'Right' and RL_2 == 'Right':
            return 'Same'
        elif RL_1 == 'Left' and RL_2 == 'Left':
            return 'Same'
        elif RL_1 == 'Left' and RL_2 == 'Right':
            return 'Opp'
        elif RL_1 == 'Right' and RL_2 == 'Left':
            return 'Opp'           
        else:
            return 'Non'
    except:
        return np.nan

def add_avg_pair_select_idx(xt, yt):
    if np.isnan(xt).any() or np.isnan(yt).any():
        return np.nan
    else:
        idx = add_pair_select_idx(np.mean(xt), np.mean(yt))
        return idx

def calc_pair_selectivity_detailed(RL_1, RL_2):
    try:
        if not RL_1 in ['Non','Right','Left','Mixed']:
            return np.nan
        elif not RL_2 in ['Non','Right','Left','Mixed']:
            return np.nan
        elif RL_1 == 'Right' and RL_2 == 'Right':
            return 'R-R'
        elif RL_1 == 'Left' and RL_2 == 'Left':
            return 'L-L'
        elif RL_1 == 'Left' and RL_2 == 'Right':
            return 'R-L'
        elif RL_1 == 'Right' and RL_2 == 'Left':
            return 'R-L'
        elif RL_1 == 'Non' and RL_2 == 'Non':
            return 'N-N'
        elif RL_1 == 'Non' and RL_2 == 'Right':
            return 'R-N'
        elif RL_1 == 'Non' and RL_2 == 'Left':
            return 'L-N'
        elif RL_2 == 'Non' and RL_1 == 'Right':
            return 'R-N'
        elif RL_2 == 'Non' and RL_1 == 'Left':
            return 'L-N'                    
        else:
            return np.nan
    except:
        return np.nan

def calc_pair_selectivity(RL_1, RL_2):
    try:
        if not RL_1 in ['Non','Right','Left','Mixed']:
            return np.nan
        elif not RL_2 in ['Non','Right','Left','Mixed']:
            return np.nan
        elif RL_1 == 'Right' and RL_2 == 'Right':
            return 'Same'
        elif RL_1 == 'Left' and RL_2 == 'Left':
            return 'Same'
        else:
            return 'Different'
    except:
        return np.nan

def add_RL_pair_types(row, neuron_df):
    RL_1 = row.source_selectivity
    RL_2 = row.target_selectivity
    return calc_pair_selectivity(RL_1, RL_2)
def add_RL_pair_types_opp(row, neuron_df):
    RL_1 = row.source_selectivity
    RL_2 = row.target_selectivity
    return calc_pair_selectivity_opp(RL_1, RL_2)
def add_RL_pair_types_detailed(row, neuron_df):
    RL_1 = row.source_selectivity
    RL_2 = row.target_selectivity
    return calc_pair_selectivity_detailed(RL_1, RL_2)
def add_ori_pair_types_old(pref_ori_diff, neuron_df):
    try:
        if pref_ori_diff <= -45:
            return 'Opp'
        elif pref_ori_diff > -45:
            return 'Same'
        else:
            return np.nan
    except:
        return np.nan
def add_ori_pair_types(row, neuron_df):
    pref_ori_diff = np.abs(row['source_select_idx']-row['target_select_idx'])
    pref_ori_diff = -np.min([pref_ori_diff, 180-pref_ori_diff])
    try:
        if pref_ori_diff <= -45:
            return 'Opp'
        elif pref_ori_diff > -45:
            return 'Same'
        else:
            return np.nan
    except:
        return np.nan
def calc_ori_pair_sel(select_diff):
    if select_diff > 45:
        return 'Opp'
    elif select_diff <= 45:
        return 'Same'
    else:
        return 'unknown'
def add_ori_pair_idx_old(row):
    RL_diff = np.abs(row['source_select_idx']-row['target_select_idx'])
    return -np.min([RL_diff, 180-RL_diff])
def add_ori_pair_idx(row, avg = None):
    RL_diff = np.abs(row['source_select_idx']-row['target_select_idx'])
    RL_diff = np.min([RL_diff, 180-RL_diff])
    if avg == 'geom':
        mag = scipy.stats.mstats.gmean([row['source_oritunsel'], row['target_oritunsel']])
    elif avg == 'mean':
        mag = np.average([row['source_oritunsel'], row['target_oritunsel']])
    elif avg == None:
        mag = 1
    pair_select_idx = mag * (45 - RL_diff) / 45
    return pair_select_idx
def add_RL_diff(row, select_idx = 'RL_select_idx_ROC'):
    RL_diff = np.abs(row['source_select_idx']-row['target_select_idx'])
    if select_idx == 'oripeaksel':
        return np.min([RL_diff, 180-RL_diff])
    elif select_idx == 'dirpeaksel':
        return np.min([RL_diff, 360-RL_diff])
    else:
        return RL_diff
def add_RL_sim(row): 
    return 1-np.abs(row['source_select']-row['target_select'])
def add_psd_areas(row, syn_df):
    syns = syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)]
    return syns.psd_area.values
def add_avg_psd_area(row, syn_df):
    syns = syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)]
    return np.average(syns.psd_area.values)
def add_connector_ids(row, syn_df):
    mySyn = syn_df[syn_df.source==row.source]
    mySyn = mySyn[mySyn.target==row.target]
    return mySyn.connector_id.values

def list2pairs(psd_areas):
    if len(psd_areas) < 2:
        return None
    elif len(psd_areas) == 2:
        return psd_areas
    else:
        pairs = None
        for i in range(len(psd_areas)):
            for j in range(i):
                myPair = np.array([psd_areas[i], psd_areas[j]])
                if pairs is not None:
                    pairs = np.vstack((pairs, myPair))
                else:
                    pairs = myPair
        return pairs

def psd_pairs_from_grouped_syns(psd_areas_array):
    pairs_array = None
    for psd_areas in psd_areas_array:
        myPairs = list2pairs(psd_areas)
        if myPairs is not None:
            if pairs_array is not None:
                pairs_array = np.vstack([pairs_array, myPairs])
            else:
                pairs_array = myPairs
    return pairs_array

def calc_triad_type(source_A, source_B, target):
    triad_type = None
    if target == 'pyramidal':
        if source_A == 'pyramidal' and source_B == 'pyramidal':
            triad_type = 'EE-E'
        elif source_A == 'non pyramidal' and source_B == 'non pyramidal':
            triad_type = 'II-E'
        else:
            triad_type = 'EI-E'
    elif target == 'non pyramidal':
        if source_A == 'pyramidal' and source_B == 'pyramidal':
            triad_type = 'EE-I'
        elif source_A == 'non pyramidal' and source_B == 'non pyramidal':
            triad_type = 'II-I'
        else:
            triad_type = 'EI-I'
    return triad_type
def compare_epochs(epoch_A, epoch_B, adj = False):
    if not adj:
        if epoch_A == epoch_B:
            return 'Same'
        else:
            return 'Diff'
    else:
        epochs = np.array(['ITIbefore', 'cueEarly', 'cueLate', 'delay','turn','ITI'])
        epoch_idx_A = np.where(epochs == epoch_A)[0]
        epoch_idx_B = np.where(epochs == epoch_B)[0]
        if np.abs(epoch_idx_A-epoch_idx_B) <= 1:
            return 'Same'
        else:
            return 'Diff'

def add_cable_overlaps(cn_DF, max_dist = '5 microns', dendrite_type = 'dendrite', proximal_r=64):
    labels = pymaid.get_label_list()
    sources = pymaid.get_neuron(np.unique(cn_DF.source.values))
    source_axons = cAPI.neuronList_to_neuronParts(sources,labels,parts='axon', proximal_r=proximal_r)
    source_axons_res = source_axons.resample('.1 micron', inplace=False)
    navis.smooth_skeleton(source_axons_res, window=10, to_smooth=['x', 'y', 'z'], inplace=True) # smooth to 1 um
    targets = pymaid.get_neuron(np.unique(cn_DF.target.values))
    target_dendrites = cAPI.neuronList_to_neuronParts(targets,labels,parts=dendrite_type, proximal_r=proximal_r)
    
    # parts can be ['dendrite', 'proximal', 'apical', 'basal']
    target_dendrites_res = target_dendrites.resample('.1 micron', inplace=False)
    navis.smooth_skeleton(target_dendrites_res, window=10, to_smooth=['x', 'y', 'z'], inplace=True) # smooth to 1 um
    # navis.cable_overlap: method='forward' is from axon perspective, method='reverse' is from dendrite
    overlaps = navis.cable_overlap(source_axons_res, target_dendrites_res, dist=max_dist, method='forward')
    #overlaps = navis.cable_overlap(sources, targets, dist=max_dist, method='avg') # control without axon/dendrite cuts

    cn_DF['cable_overlap'] = cn_DF.apply (lambda row: overlaps[str(row.target)].loc[str(row.source)], axis=1)/1000 #units in microns
    def div(a,b):
        if b == 0:
            return np.nan
        else:
            return a/b 
    cn_DF['syn_den'] = cn_DF.apply (lambda row: div(row['syn_count'],row['cable_overlap']), axis=1)
    #cn_DF['cn_strength'] = cn_DF.apply (lambda row: row['syn_den']*row['avg_psd_area'], axis=1)
    #cn_DF['cn_strength'] = cn_DF['cn_strength']/np.nanmean(cn_DF['cn_strength'].values)
    return cn_DF

def classify_dendrite_target_old(row):
    soma_r = 20
    try:
        if row['target_type'] == 'pyramidal':
            if row['target_loc'] == 'unknown':
                return 'unknown'
            elif row['syn_PSsoma_pathdist'] < soma_r:
                return 'proximal'    
            elif row['connector_loc'][1] < row['target_loc'][1]:
                return 'apical'
            else:
                return 'basal'
        else:
            return 'unknown'
    except: 
        print('ERROR: source:target %i:%i has no dendrite target type info' % (row['source'],row['target']))
        return 'unknown'

def classify_dendrite_target(row, labels, proximal_r = 64):
    try:
        if row['syn_PSsoma_pathdist'] <= proximal_r:
            return 'proximal'
        elif row['syn_PSsoma_pathdist'] > proximal_r:
            if row['target_type'] == 'pyramidal':
                myTarget_id = row.target
                myTarget = pymaid.get_neuron(myTarget_id)
                myTargetNode = pymaid.get_connector_details(row.connector_id).postsynaptic_to_node.values[0][0]
                myApLbl = cAPI.get_tag_id(myTarget,labels,'apical dendrite')
                if myApLbl is not None:
                    if navis.distal_to(myTarget, myTargetNode, myApLbl):
                        return 'apical'
                    else:
                        return 'basal'
                else:
                    return 'basal'
            else:
                return 'non-pyramidal'
        else:
            return 'unknown'
    except:
        return 'unknown'

def classify_dendrite_target_2way(row, labels, proximal_r = 64):
    try:
        if row['syn_PSsoma_pathdist'] <= proximal_r:
            return 'proximal'  
        elif row['syn_PSsoma_pathdist'] > proximal_r:
            return 'distal'
        else:
            return 'unknown'
    except:
        return 'unknown'

def calc_syn_freq_totals(pot_cns, shuf = False, bootstrap = False):
    if bootstrap:
        pot_cns = pot_cns.sample(frac=1, replace=True)
    elif shuf:
        pot_cns['source_select_idx_new_shuf'] = pot_cns.sample(frac=1).source_select_idx_new.values
        pot_cns['target_select_idx_new_shuf'] = pot_cns.sample(frac=1).target_select_idx_new.values
        #pot_cns['pair_select_sign'] = pot_cns.sample(frac=1).pair_select_sign.values
        pot_cns['pair_select_idx_new'] =  pot_cns.apply (lambda row: add_pair_select_idx(row['source_select_idx_new_shuf'], row['target_select_idx_new_shuf']), axis=1)
        pot_cns['pair_select_sign'] = ['co' if i>=0 else 'anti' for i in pot_cns.pair_select_idx_new]
    num_co = len(pot_cns[pot_cns.pair_select_sign == 'co'])
    num_anti = len(pot_cns[pot_cns.pair_select_sign == 'anti'])
    overlap_co = np.sum(pot_cns[pot_cns.pair_select_sign == 'co'].cable_overlap)
    overlap_anti = np.sum(pot_cns[pot_cns.pair_select_sign == 'anti'].cable_overlap)
    count_co = np.sum(pot_cns[pot_cns.pair_select_sign == 'co'].syn_count)
    count_anti = np.sum(pot_cns[pot_cns.pair_select_sign == 'anti'].syn_count)
    freq_co = np.sum(pot_cns[pot_cns.pair_select_sign == 'co'].syn_count)/np.sum(pot_cns[pot_cns.pair_select_sign == 'co'].cable_overlap)
    freq_anti = np.sum(pot_cns[pot_cns.pair_select_sign == 'anti'].syn_count)/np.sum(pot_cns[pot_cns.pair_select_sign == 'anti'].cable_overlap)
    freq_all = np.sum(pot_cns.syn_count)/np.sum(pot_cns.cable_overlap)
    freq_diff = freq_co - freq_anti
    df = pd.DataFrame({'num_co':num_co, 'num_anti':num_anti,'freq_co':freq_co, 'freq_anti':freq_anti,'freq_all':freq_all, 'freq_diff':freq_diff,
        'overlap_co':overlap_co, 'overlap_anti':overlap_anti, 'count_co':count_co, 'count_anti':count_anti},index=[0])
    return df

def calc_syn_freq_totals_epochs(pot_cns, shuf = False, bootstrap = False):
    if bootstrap:
        pot_cns = pot_cns.sample(frac=1, replace=True)
    elif shuf:
        pot_cns['pair_epoch'] = pot_cns.sample(frac=1).pair_epoch.values
    freq_same = np.sum(pot_cns[pot_cns.pair_epoch == 'Same'].syn_count)/np.sum(pot_cns[pot_cns.pair_epoch == 'Same'].cable_overlap)
    freq_diff = np.sum(pot_cns[pot_cns.pair_epoch == 'Diff'].syn_count)/np.sum(pot_cns[pot_cns.pair_epoch == 'Diff'].cable_overlap)
    freq_all = np.sum(pot_cns.syn_count)/np.sum(pot_cns.cable_overlap)
    freq_sub = freq_same - freq_diff
    df = pd.DataFrame({'freq_same':freq_same, 'freq_diff':freq_diff,'freq_all':freq_all, 'freq_sub':freq_sub},index=[0])
    return df


def calc_avg_pair_select(pot_cns, shuf = False, bootstrap = False):
    if bootstrap:
        pot_cns = pot_cns.sample(frac=1, replace=True)
    elif shuf:
        pot_cns['pair_select_idx'] = pot_cns.sample(frac=1).pair_select_idx.values
    select_cn = np.nanmean(pot_cns[pot_cns.connected == True].pair_select_idx)
    select_un = np.nanmean(pot_cns[pot_cns.connected == False].pair_select_idx)
    select_diff = select_cn - select_un
    df = pd.DataFrame({'select_cn':select_cn, 'select_un':select_un,'select_diff':select_diff},index=[0])
    
    return df