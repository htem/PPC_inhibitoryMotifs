
import pandas as pd
import pymaid
import numpy as np
import seaborn as sns
import pymaid
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import itertools
import networkx as nx
import time
import navis
import math

def query_cell_info(myNeuronDf,myNeuron,labels,parseLayers=True):
    def get_cell_layer(somaLoc):
        volumes = ['L1','L23','L5','L6',]
        for idx, vol in enumerate(volumes):
            # go from top to bottom bc volumes are curved up and convex hull is only approximation
            if pymaid.in_volume(somaLoc,vol)[0]:
                return vol
        return 'unknown'
    skelID = int(myNeuron.skeleton_id)
    myNeuronDf['skeleton_id'] = skelID
    myNeuronDf['type'] = get_cell_type(skelID)
    #myNeuronDf['type_w_ap'] = get_cell_type(skelID, apicals=True)
    soma_id = pymaid.has_soma(skelID, return_ids=True)
    if len(soma_id[int(skelID)]) == 1:
        myNeuronDf['has_soma'] = True
        myNodeLoc = pymaid.get_node_location(soma_id[int(skelID)][0])
        somaLoc=np.asarray([myNodeLoc.x[0],myNodeLoc.y[0],myNodeLoc.z[0]])
        myNeuronDf['soma_loc'] = somaLoc

        if parseLayers:
            myNeuronDf['layer'] = get_cell_layer(np.asarray([somaLoc]))
        else:
            myNeuronDf['layer'] = 'unknown'
    else:

        myNeuronDf['has_soma'] = False
        myNeuronDf['layer'] = 'unknown'
        myNeuronDf['soma_loc'] = [np.nan, np.nan, np.nan]
    return(myNeuronDf)

def get_cell_type(skel_ids, apicals = False):
    def parse_celltype_annot(annot):
        myNeuronType = 'unknown'
        if bool(annot):
            if 'pyramidal' in annot and not 'non pyramidal' in annot:
                myNeuronType = 'pyramidal'
            elif 'non pyramidal' in annot:
                myNeuronType = 'non pyramidal'
            elif 'apical dendrite' in annot:
                if apicals:
                    myNeuronType = 'apical dendrite' # ATK 211020
                else:
                    myNeuronType = 'pyramidal'
        return myNeuronType

    myNeuronTypes = []
    annotations = pymaid.get_annotations(skel_ids)
    if type(skel_ids) is list or type(skel_ids) is np.ndarray:
        for skelID in skel_ids:
            if str(skelID) in annotations.keys():
                myNeuronType = parse_celltype_annot(annotations[str(skelID)])
            else:
                myNeuronType = 'unknown'
            myNeuronTypes.append(myNeuronType)
        return myNeuronTypes
    else:
        if str(skel_ids) in annotations.keys():
            myNeuronType = parse_celltype_annot(annotations[str(skel_ids)])
        else:
            myNeuronType = 'unknown'
        return myNeuronType

def query_axon(myNeuronDf,myCellID,myNeuron,myLabels):

    myelinates = np.nan
    firstMynDist = np.nan
    fracMyn = np.nan
    numChanSyn = np.nan
    numAxonSyn = np.nan
    numTrunkSyn = np.nan
    numCollateralSyn = np.nan
    axonLen = np.nan
    trunkLen = np.nan
    collateralLen = np.nan
    synDensity_collat = np.nan
    synDensity_trunk = np.nan

    (myAxon, myAxonTrunk, myAxonCollaterals, myMyelination) = get_axon_components(myNeuron, myLabels)
    myCollats_df = get_collat_df(myAxonCollaterals, myNeuron)

    numAxonSyn = count_outgoing_synapses(myAxon)
    if myAxon is not None:
        numChanSyn = count_incoming_synapses(myAxonTrunk)
        numTrunkSyn = count_outgoing_synapses(myAxonTrunk)
        numCollateralSyn = count_outgoing_synapses(myAxonCollaterals)

        axonLen = myAxon.cable_length / 1000
        trunkLen = myAxonTrunk.cable_length / 1000 #Units in um
        collateralLen = myAxonCollaterals.cable_length / 1000
        myelinationLen = myMyelination.cable_length / 1000 

        if trunkLen > 100 and numTrunkSyn > 0:
            synDensity_trunk = numTrunkSyn/trunkLen
        if collateralLen > 50 and numCollateralSyn > 0:
            synDensity_collat = numCollateralSyn/collateralLen

        # Find first myelination and measure pathlength from soma
        # Get myelination labels
        myMyn = myLabels[ myLabels.tag == 'myelinates']
        myUnmyn = myLabels[ myLabels.tag == 'unmyelinate']
        mynDists = []
        for idx,i in enumerate(myMyn.node_id):
            mynDists.append(navis.dist_between(myNeuron,myNeuron.root,myMyn.node_id.values[idx]))
        if len(mynDists) > 0:
            myelinates = True
            firstMynDist = min(mynDists)/1000 # in microns
        elif trunkLen > 150:
            myelinates = False
            firstMynDist = np.nan
        else:
            myelinates = np.nan
            firstMynDist = np.nan

        # Calculate proportion of myelinated pathlength
        if myelinates and trunkLen>0:
            fracMyn = myelinationLen/trunkLen
        else:
            fracMyn = np.nan

    myNeuronDf['num_chandelier_synapses'] = numChanSyn
    myNeuronDf['num_axon_trunk_synapses'] = numTrunkSyn
    myNeuronDf['num_axon_collateral_synapses'] = numCollateralSyn
    myNeuronDf['num_axon_synapses'] = numAxonSyn
    myNeuronDf['axon_len'] = axonLen
    myNeuronDf['axon_len_trunk'] = trunkLen
    myNeuronDf['axon_len_collats'] = collateralLen
    myNeuronDf['myelinates'] = myelinates
    myNeuronDf['first_myn_dist'] = firstMynDist
    myNeuronDf['frac_myn'] = fracMyn
    myNeuronDf['syn_density_collat'] = synDensity_collat 
    myNeuronDf['syn_density_trunk'] =synDensity_trunk

    if len(myCollats_df) > 0:
        myNeuronDf['num_collats'] = len(myCollats_df)
        myNeuronDf['collat_density'] = len(myCollats_df)/trunkLen
        myNeuronDf['collat_lengths'] = myCollats_df['cable_length']
        myNeuronDf['collat_depths'] = myCollats_df['collat_depths']
        myNeuronDf['collat_angles'] = myCollats_df['collat_angles']
        myNeuronDf['synapse_ids'] = myCollats_df['synapse_ids'] 
        myNeuronDf['collat_syn_count'] = myCollats_df['syn_count']
        myNeuronDf['collat_syn_density'] = myCollats_df['syn_density'] 
    else:
         myNeuronDf['num_collats'] = 0
    return(myNeuronDf)

def get_axon_components(myNeuron, myLabels):
    # Define axon based on "axon" tag and count synapses
    myAxnLbl = myLabels[ myLabels.tag == 'axon']

    if len(myAxnLbl) < 1:
        print('WARNING: no axon tag on skel %s' % myNeuron.skeleton_id)
        return (None, None, None, None)
    elif len(myAxnLbl) == 1:
        myAxnStartID = myAxnLbl.node_id.values
    else:
        print('WARNING: more than one axon tag on skel %s' % myNeuron.skeleton_id)
        myAxnStartID = myAxnLbl.node_id.values[0]

    #myAxon = navis.cut_neuron(myNeuron,int(myAxnStartID),ret='distal')
    myAxon = navis.cut_skeleton(myNeuron,int(myAxnStartID),ret='distal') # Note, this is for dev branch of navis 210921, master branch has function neamed "cut_neuron"
    # Separate axon trunk

    axonTrunkNodes = np.asarray(myAxon.nodes.node_id[myAxon.nodes.radius.values>0])
    axonCollateralNodes = np.asarray(myAxon.nodes.node_id[myAxon.nodes.radius.values<=0])
    myAxonTrunk = navis.subset_neuron(myAxon,axonTrunkNodes)
    myAxonCollaterals = navis.subset_neuron(myAxon,axonCollateralNodes)

    myMyn = myLabels[ myLabels.tag == 'myelinates']
    myUnmyn = myLabels[ myLabels.tag == 'unmyelinates']
    myMynNodeIDs = find_myelination(myNeuron,myMyn,myUnmyn)
    myMyelination = navis.subset_neuron(myNeuron,np.asarray(myMynNodeIDs))

    axonNodes = np.asarray(myAxon.nodes.node_id) # hack needed to convert to CatmaidNeuron (instead of navis Neuronlist)
    myAxon = navis.subset_neuron(myAxon,axonNodes)

    return myAxon, myAxonTrunk, myAxonCollaterals, myMyelination

def get_collat_df(myAxonCollaterals, myNeuron):
    def check_if_branch(myCollat):
        myCollatPruned = navis.prune_by_strahler(myCollat, to_prune=slice(0, -1)) # only retain highest strahler order
        myNodeTags = pymaid.get_node_tags(myCollatPruned.nodes.node_id.values, 'NODE')
        myTags = [item for sublist in [myNodeTags[a] for a in myNodeTags.keys()] for item in sublist]
        if 'not a branch' in myTags:
            return True
        else:
            return False
    def calc_angle(myCollat, length = '20 microns'):
        collat_trunc = navis.prune_at_depth(myCollat,'20 um', inplace = False)
        root_loc = collat_trunc.nodes.iloc[-1][['x','y','z']].values
        end_loc = collat_trunc.nodes.iloc[0][['x','y','z']].values
        delta_loc = end_loc-root_loc
        dr = np.abs(math.sqrt(delta_loc[0]**2+delta_loc[2]**2))
        dz = delta_loc[1]
        angle = math.degrees(math.atan(dr/dz))
        if angle < 0:
            angle = 180 + angle # want between 0 and 180 
        return angle
    if myAxonCollaterals is not None:
        myCollats = navis.break_fragments(myAxonCollaterals, labels_only=False, min_size=3)
        if len(myCollats) == 0:
            return pd.DataFrame()
        else:
            collat_idxs = range(len(myCollats))
            myCollats_df = pd.DataFrame({'n_nodes':myCollats.n_nodes,'cable_length':myCollats.cable_length/1000})
            myCollatRootNodes = [myCollats[i].root[0] for i in collat_idxs]
            myCollats_df['collat_root_node'] = myCollatRootNodes 
            myCollatBranchNodes = [myNeuron.nodes[myNeuron.nodes.node_id == myCollatRootNodes[i]].parent_id.values[0] for i in collat_idxs]
            myCollats_df['collat_branch_node'] = myCollatBranchNodes
            myCollats_df['not_a_branch'] = [check_if_branch(myCollats[i]) for i in collat_idxs]
            if myNeuron.soma is None: 
                myCollats_df['collat_depths'] = [np.nan for i in collat_idxs]
            else:
                myCollats_df['collat_depths'] = np.array([navis.dist_between(myNeuron, myCollatBranchNodes[i], myNeuron.soma) for i in collat_idxs])/1000
            myCollats_df['collat_angles'] = [calc_angle(myCollats[i]) for i in collat_idxs]
            myCollats_df['synapse_ids'] = [myCollats[i].presynapses.connector_id.values for i in collat_idxs]
            myCollats_df['syn_count'] = [myCollats[i].n_presynapses for i in collat_idxs]
            myCollats_df['syn_density'] = myCollats_df['syn_count'] / myCollats_df['cable_length'] 
        return myCollats_df
    else: 
        return pd.DataFrame()

# needs check
def find_myelination(myNeuron,myMynTags,unMynTags):
    myMynNodeIDs = []
    myUnMynNodeIDs = []
    for myNodeID in unMynTags.node_id.values:
        # walk upstream until you find a "myelinates" tag
        loop = True
        while loop == True:
           myMynNodeIDs.append(myNodeID)
           myNodeID=list(myNeuron.graph.successors(myNodeID))[0]
           if len(list(myNeuron.graph.successors(myNodeID))) > 1:
              print('multiple parents')
              loop = False
              return([])
           if myNodeID in myMynTags.node_id.values:
              myUnMynNodeIDs.append(myNodeID)
              loop = False
           elif myNodeID == myNeuron.root:
              print('hit root node')
              loop = False
              return([])
    # need to add case if end is myelinated, ie number of myelin tags more than unmyn
    if len(myMynTags)>len(unMynTags):
        print('end is myelinated for neuron %s (code needs update)' % myNeuron.skeleton_id)
    return( myMynNodeIDs )



def count_incoming_synapses(myNeuron):
    try:
        myCns = pymaid.cn_table_from_connectors(myNeuron)
        myCnsUp = myCns[ myCns.relation == 'upstream' ]
        return len(myCnsUp.index)
    except:
        return None

def count_outgoing_synapses(myNeuron):
    try:
        myCns = pymaid.cn_table_from_connectors(myNeuron)
        myCnsDown = myCns[ myCns.relation == 'downstream' ]
        return len(myCnsDown.index)
    except:
        return np.nan

def get_old_cid_from_skelID(skelID):
    cid = 0
    annotations = pymaid.get_annotations(skelID)
    annotations[str(skelID)]
    result = [i for i in annotations[str(skelID)] if i.startswith('Matched Neuron ')]
    if len(result) > 0:
        cid = result[0][-3:]
    return cid

def get_skelID_from_old_cid(cid):
    try:
        if cid > 100:
            myNeuron = pymaid.find_neurons(annotations="Matched Neuron %d" % cid)
            return myNeuron[0].skeleton_id
        else: # to add leading zeros (convention for now)
            myNeuron = pymaid.find_neurons(annotations="Matched Neuron 0%d" % cid)
            return myNeuron[0].skeleton_id
    except:
        print("No skeleton found with annotation Matched Neuron %d!" % cid)
        return None

def get_cid_list_from_skelID(skelList):
    cids = []
    for skelID in skelList:
        cids.append(get_cid_from_skelID(skelID))
    return cids

def get_cid_from_skelID(skelID, annotTag='new matched neuron'):
    cid = np.nan
    annotations = pymaid.get_annotations(skelID)
    #annotations[str(skelID)]
    #annotTag = 'new matched neuron '
    if str(skelID) in annotations.keys():
        result = [i for i in annotations[str(skelID)] if i.startswith(annotTag)]
        if len(result) > 0:
            if len(result) > 1:
                print('WARNING: more than one matched neuron num')
            #cid = result[0][-4:]
            try:
                cid = float(result[0][len(annotTag):])
            except:
                print('Warning - non numerical cid')
    return cid

def get_skelID_from_cid(cid, dataset = 'PPC'):
    try:
        if dataset == 'PPC':
            rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff')
            myNeuron = pymaid.find_neurons(annotations="new matched neuron %d" % cid,partial_match=False)
            return myNeuron[0].skeleton_id
        elif dataset == 'V1':
            rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
                api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff', project_id=31)
            myNeuron = pymaid.find_neurons(annotations="matched neuron %d" % cid,partial_match=False)
            return myNeuron[0].skeleton_id
        #else: # to add leading zeros (convention for now)
        #    myNeuron = pymaid.find_neurons(annotations="Matched Neuron 0%d" % cid)
        #    return myNeuron[0].skeleton_id
    except:
        print("No skeleton found with annotation new matched neuron %d!" % cid)
        return None

def get_2P_ROIs(myNeuronDf,myCellID,myNeuron,myLabels,mySessions):
    myAnnots = pymaid.get_annotations(myNeuron.skeleton_id)
    myAnnots = myAnnots[str(myNeuron.skeleton_id)]
    sessions_ROI = np.empty(len(mySessions),dtype=np.int)
    for idx,session in enumerate(mySessions):
        session_tag = [item for item in myAnnots if item.startswith(session)]
        if session_tag:
            #sessions_ROI.append(session_tag[0][13:])
            sessions_ROI[idx]=int(session_tag[0][13:])
        else:
            #sessions_ROI.append([])
            sessions_ROI[idx]=-1
    myNeuronDf['sessions_ROI'] = sessions_ROI
    return myNeuronDf

def get_corr_quality(skel_ids):
    def parse_corr_annot(annot):
        myNeuronType = 'unknown'
        if bool(annot):
            if 'corr good' in annot:
                myNeuronType = 'good'
            elif 'corr okay' in annot:
                myNeuronType = 'okay'
            elif 'corr bad' in annot:
                myNeuronType = 'bad'
        return myNeuronType
    myNeuronTypes = []
    annotations = pymaid.get_annotations(skel_ids)
    if type(skel_ids) is list or type(skel_ids) is np.ndarray:
        for skelID in skel_ids:
            if str(skelID) in annotations.keys():
                myNeuronType = parse_corr_annot(annotations[str(skelID)])
            else:
                myNeuronType = 'unknown'
            myNeuronTypes.append(myNeuronType)
        return myNeuronTypes
    else:
        if str(skel_ids) in annotations.keys():
            myNeuronType = parse_corr_annot(annotations[str(skel_ids)])
        else:
            myNeuronType = 'unknown'
        return myNeuronType

def add_tracing_status(psp_df, ps_annot):
    def get_tracing_status(row, ps_annot):
        if row['has_soma'] == True:
            return 'has_soma'
        elif row['skeleton_id'] in ps_annot.keys():
                annotations = ps_annot[row['skeleton_id']]
                if 'soma outside volume' in annotations:
                    return 'soma_outside_volume'
                elif 'uncertain continuation' in annotations:
                    return 'uncertain_continuation'
                elif 'glia' in annotations:
                    return 'glia'
                else:
                    return 'unknown'
        else:
            return 'unknown'
    psp_df['tracing_status'] = psp_df.apply (lambda row:
        get_tracing_status(row, ps_annot), axis=1)
    return psp_df

def add_axon_type(synapse_df):
    def get_axon_type(row):
        source_type = row['source_type']
        if source_type == 'pyramidal':
            connector_id = row['connector_id']
            axon_node = pymaid.get_connector_details(connector_id).presynaptic_to_node.values[0]
            axon_rad = pymaid.find_nodes(node_ids=axon_node).radius.values[0]
            if axon_rad > 0:
                return 'trunk'
            else:
                return 'collateral'
        else:
            return 'unknown'

    synapse_df['axon_type'] = synapse_df.apply (lambda row:
        get_axon_type(row), axis=1)
    return synapse_df

# Add physiology metrics to tracingDataframe
def query_trialAvgActivity(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,Ca_trial_means):
    my_act_wL = None
    my_act_wL_sum = np.zeros(63)
    my_act_bR = None
    my_act_bR_sum = np.zeros(63)
    num_active_sessions = 0
    sessions_ROI = myNeuronDf['sessions_ROI']
    for session_idx,session in enumerate(mySessions):
            if sessions_ROI[session_idx]>-1:
                #print(Ca_trial_means[session]['wL_trials'].shape)
                my_act_wL_sum += Ca_trial_means[session]['wL_trials'][:,int(sessions_ROI[session_idx])]
                my_act_bR_sum += Ca_trial_means[session]['bR_trials'][:,int(sessions_ROI[session_idx])]
                num_active_sessions += 1
    if num_active_sessions > 0:
        my_act_wL = my_act_wL_sum/num_active_sessions
        my_act_bR = my_act_bR_sum/num_active_sessions
    else:
        my_act_wL = np.nan
        my_act_bR = np.nan
    myNeuronDf['Ca_trial_mean_wL'] = my_act_wL
    myNeuronDf['Ca_trial_mean_bR'] = my_act_bR
    myNeuronDf['num_active_sessions'] = num_active_sessions

def query_trialAvgMI(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,choiceMI):
    my_act_wL = None
    my_act_wL_sum = np.zeros(63)
    my_act_bR = None
    my_act_bR_sum = np.zeros(63)
    num_active_sessions = 0
    sessions_ROI = myNeuronDf['sessions_ROI']
    for session_idx,session in enumerate(mySessions):
            if sessions_ROI[session_idx]>-1:
                my_MI_sum = choiceMI[session][:,int(sessions_ROI[session_idx])]
                num_active_sessions += 1
    if num_active_sessions > 0:
        my_MI = my_MI_sum/num_active_sessions
    else:
        my_MI = np.nan
    myNeuronDf['MI_trial_avg'] = my_MI

def query_trial_snr(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,trial_snr):
    my_trial_snr_wL = None
    my_trial_snr_wL_sum = 0
    my_trial_snr_bR = None
    my_trial_snr_bR_sum = 0
    num_active_sessions = 0
    sessions_ROI = myNeuronDf['sessions_ROI']
    for session_idx,session in enumerate(mySessions):
            if sessions_ROI[session_idx]>-1:
                my_trial_snr_wL_sum += trial_snr[session]['wL_trials'][0,int(sessions_ROI[session_idx])]
                my_trial_snr_bR_sum += trial_snr[session]['bR_trials'][0,int(sessions_ROI[session_idx])]
                num_active_sessions += 1
    if num_active_sessions > 0:
        my_trial_snr_wL = my_trial_snr_wL_sum/num_active_sessions
        my_trial_snr_bR = my_trial_snr_bR_sum/num_active_sessions
    else:
        my_trial_snr_wL = np.nan
        my_trial_snr_bR = np.nan
    myNeuronDf['trial_snr_wL'] = my_trial_snr_wL
    myNeuronDf['trial_snr_bR'] = my_trial_snr_bR
    myNeuronDf['trial_snr_max'] = np.maximum(my_trial_snr_wL,my_trial_snr_bR)

    # Query cell type to myNeuronDf
def query_BS_inputs(myNeuronDf,myCellID,myNeuron,myLabels):

    skel_id = myNeuron.skeleton_id

    # Define soma based on root node and count synapses
    mySoma = navis.cut_neuron(myNeuron,myNeuron.root,ret='proximal')
    try:
        mySomaCns = pymaid.cn_table_from_connectors(mySoma)
        mySomaCnsUp = mySomaCns[ mySomaCns.relation == 'upstream' ]
        numSomaCnsUp = len(mySomaCnsUp.index)
    except:
        numSomaCnsUp = np.nan

    myNeuronDf['BC_inputs'] = numSomaCnsUp

    return(myNeuronDf)

    # Add chandelier cell input count to myNeuronDf
def query_CC_inputs(myNeuronDf,myCellID,myNeuron,myLabels):

    skel_id = myNeuron.skeleton_id
    # Define axon based on "axon" tag and count synapses
    myAxnLbl = myLabels[ myLabels.tag == 'axon']
    if len(myAxnLbl) > 0:
        try:
            myAxnStartID = myAxnLbl.node_id.values
            myAxonFrag = navis.cut_neuron(myNeuron,myAxnStartID,ret='distal')
            myAxonCns = pymaid.cn_table_from_connectors(myAxonFrag)
            myAxonCnsUp = myAxonCns[ myAxonCns.relation == 'upstream' ]
            myAxonCnsDwn = myAxonCns[ myAxonCns.relation == 'downstream' ]
            numAxonCnsUp = len(myAxonCnsUp.index)
        except:
            numAxonCnsUp = np.nan
    else:
        numAxonCnsUp = np.nan

    myNeuronDf['CC_inputs'] = numAxonCnsUp

    return(myNeuronDf)

def plotNeuronByCid(myCellID):
    mySkelID = ID_from_cid(myCellID)
    myNeuron = pymaid.get_neuron(mySkelID)
    fig,ax = pymaid.plot2d(myNeuron,connectors=True,linewidth=1.5,method='2d',cn_size=2,color=[0, 0, 0.5])

def plotNeuronBySkelID(mySkelID):
    myNeuron = pymaid.get_neuron(mySkelID)
    fig,ax = pymaid.plot2d(myNeuron,connectors=True,linewidth=1.5,method='2d',cn_size=2,color=[0, 0, 0.5])

def query_psp_tracing(myNeuronDf,myCellID,myNeuron,post_cn_details, PSpartners,labels):
    #post_cn = pymaid.get_connectors(myNeuron.skeleton_id,relation_type='presynaptic_to').connector_id.values
    my_post_cn = post_cn_details[post_cn_details.presynaptic_to==int(myNeuron.skeleton_id)]
    num_synout=len(post_cn_details[post_cn_details.presynaptic_to==int(myNeuron.skeleton_id)])
    #(num_synout,num_synout_soma, num_synout_outside, num_synout_uncertain,
    #        num_synout_glia, num_synout_unk) = count_synout_tracing(myNeuron.skeleton_id)
    (cn_ids_soma, cn_ids_outside, cn_ids_uncertain,
        cn_ids_glia, cn_ids_unk) = count_synout_tracing(my_post_cn, PSpartners, labels)
    myNeuronDf['synout_soma'] = cn_ids_soma
    myNeuronDf["num_synout_soma"] = len(cn_ids_soma) #num_synout_soma
    myNeuronDf['synout_outside'] = cn_ids_outside
    myNeuronDf["num_synout_outside"] = len(cn_ids_outside) #num_synout_outside
    myNeuronDf['synout_uncertain'] = cn_ids_uncertain
    myNeuronDf["num_synout_uncertain"] = len(cn_ids_uncertain) #num_synout_uncertain
    myNeuronDf['synout_glia'] = cn_ids_glia
    myNeuronDf["num_synout_glia"] = len(cn_ids_glia) #num_synout_glia
    myNeuronDf['synout_unk'] = cn_ids_unk
    myNeuronDf["num_synout_unk"] = len(cn_ids_unk) #num_synout_unk
    myNeuronDf["num_synout"] = num_synout

    return(myNeuronDf)

def count_synout_tracing(cn_details, ps_neurons, annots):
    num_synout_soma, num_synout_outside, num_synout_uncertain,num_synout_glia,num_synout_unk = (0,0,0,0,0)    
    (cn_ids_soma, cn_ids_outside, cn_ids_uncertain,cn_ids_glia, cn_ids_unk) = ([],[],[],[],[])
    if len(cn_details) > 0:
        for idx,cn_id in enumerate(cn_details.connector_id): 
            try:
                skelID = cn_details.postsynaptic_to.values[idx][0]            
                soma = ps_neurons[ps_neurons.skeleton_id == str(skelID)].soma[0]
                if soma is not None:
                    num_synout_soma+=1
                    cn_ids_soma.append(cn_id)
                elif str(skelID) in annots.keys():#bool(annotations):
                    annotations = annots[str(skelID)]
                    if 'soma outside volume' in annotations:
                        num_synout_outside+=1
                        cn_ids_outside.append(cn_id)
                    elif 'uncertain continuation' in annotations:
                        num_synout_uncertain+=1
                        cn_ids_uncertain.append(cn_id)
                    elif 'glia' in annotations:
                        num_synout_glia+=1
                        cn_ids_glia.append(cn_id)
                    else:
                        num_synout_unk+=1
                        cn_ids_unk.append(cn_id)
                else:
                    num_synout_unk+=1
                    cn_ids_unk.append(cn_id)
            except:
                print('Warning: PSP %i has unknown issue' % skelID)
                num_synout_unk+=1
                cn_ids_unk.append(cn_id)
    #return (num_synout, num_synout_soma, num_synout_outside, num_synout_uncertain, num_synout_glia,num_synout_unk)
    return (cn_ids_soma, cn_ids_outside, cn_ids_uncertain,cn_ids_glia, cn_ids_unk)

def query_psp_type(myNeuronDf,myCellID,myNeuron,labels):
    (num_psp_total, num_psp_pyr, num_psp_int, num_psp_unk) = count_psp_type(myNeuron.skeleton_id)
    myNeuronDf["num_psp_total"] = num_psp_total
    myNeuronDf["num_psp_pyr"] = num_psp_pyr
    myNeuronDf["num_psp_int"] = num_psp_int
    myNeuronDf["num_psp_unk"] = num_psp_unk
    return(myNeuronDf)

def count_psp_type(MNID):
    partnerIDs=pymaid.get_partners(MNID,min_size=1,directions=["outgoing"]).skeleton_id.values
    (num_psp_total, num_psp_pyr, num_psp_int, num_psp_unk) = (len(partnerIDs),0,0,0)
    for skelID in partnerIDs:
        annotations = pymaid.get_annotations(skelID)
        if bool(annotations):
            if 'pyramidal' in annotations[str(skelID)] and not 'non pyramidal' in annotations[skelID]:
                num_psp_pyr += 1
            elif 'non pyramidal' in annotations[str(skelID)]:
                num_psp_int += 1
            else:
                num_psp_unk += 1
    return (num_psp_total, num_psp_pyr, num_psp_int, num_psp_unk)

def calc_corr_mat(myNeuronsDf,corr_all,mySessions):
    corr_mat = np.empty([len(myNeuronsDf), len(myNeuronsDf)])
    corr_mat[:] = np.nan
    for idx_1,id_1 in enumerate(myNeuronsDf.matched_cell_ID):
        for idx_2, id_2 in enumerate(myNeuronsDf.matched_cell_ID):
            #print((id_1,id_2))
            #print((myNeuronsDf.loc[id_1].sessions_ROI,myNeuronsDf.loc[id_2].sessions_ROI))
            sessions_1_bool = np.array([len(x)>0 for x in myNeuronsDf.loc[id_1].sessions_ROI])
            sessions_2_bool = np.array([len(x)>0 for x in myNeuronsDf.loc[id_2].sessions_ROI])
            first_session_match_idx = next((i for i, j in enumerate(sessions_1_bool & sessions_2_bool) if j), None)

            if first_session_match_idx is not None:
                roi_1 = int(myNeuronsDf.loc[id_1].sessions_ROI[first_session_match_idx])
                roi_2 = int(myNeuronsDf.loc[id_2].sessions_ROI[first_session_match_idx])

                #print(roi_1,roi_2)
                if idx_1 != idx_2:
                    corr_mat[idx_1, idx_2] = corr_all[mySessions[first_session_match_idx]][roi_1,roi_2]
    return corr_mat, dist.squareform(corr_mat, checks=False)

def heatmap(mat, myNeuronsDf, cmap='jet',title='None'):
    g=sns.heatmap(mat,square=True,cmap=cmap,yticklabels=1,xticklabels=1)
    g.set_xticklabels(myNeuronsDf.matched_cell_ID.values)
    g.set_yticklabels(myNeuronsDf.matched_cell_ID.values)
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=11)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, fontsize=11)
    g.set(xlabel='cell ID', ylabel='cell ID',title=title)
    return g

def calc_adj_mat(sources, targets = None):
    #sources = myNeuronDf.skeleton_id.values
    if targets is None:
        targets = pymaid.get_partners(sources, directions=['outgoing'])
    adj_mat = pymaid.adjacency_matrix(sources, targets=targets)
    #rename_rows = dict(zip(np.asarray(myNeuronDf.skeleton_id.values,dtype=int), get_cid_list_from_skelID(myNeuronDf.skeleton_id.values)))
    #adj_mat.rename(index = rename_rows, inplace=True)
    return adj_mat

def calc_shared_psp(myNeuronsDf, adj_mat_conv):
    shared_connections_mat = np.zeros([len(myNeuronsDf), len(myNeuronsDf)])
    for idx1,source1 in enumerate(myNeuronsDf.skeleton_id):
        for idx2,source2 in enumerate(myNeuronsDf.skeleton_id):
            nz1 = np.nonzero(adj_mat_conv.loc[int(source1)].values)
            nz2 = np.nonzero(adj_mat_conv.loc[int(source2)].values)
            shared_connections_mat[idx1,idx2] = len(np.intersect1d(nz1,nz2))
        shared_connections_mat[idx1,idx1] = 0
    return shared_connections_mat

def calc_jaccard_sim(myNeuronsDf, adj_mat_conv):

    matchedNeuronList = myNeuronsDf.skeleton_id.values

    jaccard_dist = dist.pdist(adj_mat_conv, metric='jaccard')
    jaccard_dist_square = dist.squareform(jaccard_dist, force='no', checks=True)
    jaccard_dist_square[jaccard_dist_square == 0] = 1
    jaccard_sim_square = 1-jaccard_dist_square
    jaccard_sim = dist.squareform(jaccard_sim_square,checks=False)
    return jaccard_sim,jaccard_sim_square

def calc_act_corr(myNeuronsDf, corr_aligned, mySessions):

    corr_mat_combined = np.empty([len(myNeuronsDf), len(myNeuronsDf), len(mySessions)])
    sessions_overlap = np.zeros([len(myNeuronsDf), len(myNeuronsDf)])
    corr_mat_combined[:] = np.nan
    for idx_1,id_1 in enumerate(myNeuronsDf.matched_cell_ID):
        neuron_1 = myNeuronsDf[myNeuronsDf.matched_cell_ID == id_1]
        for idx_2, id_2 in enumerate(myNeuronsDf.matched_cell_ID):
            neuron_2 = myNeuronsDf[myNeuronsDf.matched_cell_ID == id_2]
            sessions_1_bool = np.array([x>-1 for x in neuron_1.sessions_ROI])
            sessions_2_bool = np.array([x>-1 for x in neuron_2.sessions_ROI])
            match_bool = np.logical_and(sessions_1_bool, sessions_2_bool)
            for session_idx,match in enumerate(match_bool[0]):
                if match:
                    roi_1 = int(neuron_1.sessions_ROI.values[0][session_idx]) # This hacky indexing needs to be fixed when i have time
                    roi_2 = int(neuron_2.sessions_ROI.values[0][session_idx])
                    if idx_1 != idx_2:
                        corr_mat_combined[idx_1, idx_2, session_idx] = corr_aligned[mySessions[session_idx]][roi_1,roi_2]
                        sessions_overlap[idx_1,idx_2] += 1
    corr_mat_average = np.nanmean(corr_mat_combined,2)
    return corr_mat_average, dist.squareform(corr_mat_average,checks=False), sessions_overlap

def calc_trial_aligned_corr(myNeuronsDf, corr_aligned, mySessions):
    corr_mat_combined = np.empty([len(myNeuronsDf), len(myNeuronsDf)])
    corr_mat_combined[:] = np.nan
    for idx_1,id_1 in enumerate(myNeuronsDf.matched_cell_ID):
        for idx_2, id_2 in enumerate(myNeuronsDf.matched_cell_ID):
            sessions_1_bool = np.array([len(x)>0 for x in myNeuronsDf.loc[id_1].sessions_ROI])
            sessions_2_bool = np.array([len(x)>0 for x in myNeuronsDf.loc[id_2].sessions_ROI])
            first_session_match_idx = next((i for i, j in enumerate(sessions_1_bool & sessions_2_bool) if j), None)

            if first_session_match_idx is not None:
                roi_1 = int(myNeuronsDf.loc[id_1].sessions_ROI[first_session_match_idx])
                roi_2 = int(myNeuronsDf.loc[id_2].sessions_ROI[first_session_match_idx])

                if idx_1 != idx_2:
                    corr_mat_combined[idx_1, idx_2] = corr_aligned[mySessions[first_session_match_idx]][roi_1,roi_2]
                    #corr_mat_wL[idx_1, idx_2] = corr_wL[mySessions[first_session_match_idx]][roi_1,roi_2]
                    #corr_mat_bR[idx_1, idx_2] = corr_bR[mySessions[first_session_match_idx]][roi_1,roi_2]
                    #if corr_mat_wL[idx_1, idx_2] > corr_mat_bR[idx_1, idx_2]:
                #        corr_mat_RLmax[idx_1,idx_2] = corr_mat_wL[idx_1, idx_2]
                #    else:
                #        corr_mat_RLmax[idx_1,idx_2] = corr_mat_bR[idx_1, idx_2]
                #else:
                #    corr_mat_RLmean[idx_1,idx_2] = corr_mat_RLmax[idx_1,idx_2] = 0
        #corr_mat_RLmean = (corr_mat_wL+corr_mat_bR)/2.0
    return corr_mat_combined, dist.squareform(corr_mat_combined,checks=False)

# convergence utility functions

def jaccard_sim_from_nx(G, source_skids):
    j = nx.jaccard_coefficient(G)
    j_df = pd.DataFrame(j)
    j_df = j_df.rename(columns={0: "source", 1: "target", 2: "weight"})
    j_G = nx.from_pandas_edgelist(j_df, edge_attr='weight')
    j_sq = nx.to_pandas_adjacency(j_G, nodelist = np.array(source_skids,dtype=int))
    #j_sim_sq_full = 1-j_sq#-np.eye(len(j_sq))
    #j_sim_sq = j_sim_sq_full.loc[np.array(source_skids,dtype=int),np.array(source_skids,dtype=int)]
    return j_sq

def bootstrap_synapses(synapse_df):
    synapse_bs = synapse_df.sample(frac=1, replace=False)
    return synapse_bs

def get_graph(source_skids, target_skids):
    synapse_df = get_synapses_between(source_skids, target_skids)
    G = nx.from_pandas_edgelist(synapse_df)
    return G

def get_cn_sim(source_skids, target_skids=None, metric='jaccard_sim', synapse_df=None, weight=None):
    if synapse_df is None:
        if target_skids is None:
            target_skids = pymaid.get_partners(source_skids,directions=['outgoing']).skeleton_id.values
        synapse_df = get_synapses_between(source_skids, target_skids)
        print('getting synapse_df')

    DiG = DiGraph_from_syn(synapse_df)
    G = DiG.to_undirected()
    G.add_nodes_from(np.array(source_skids,dtype=int))

    source_skids_int = np.array(source_skids, dtype=int)
    ebunch=[(x, y) for x, y in itertools.product(source_skids_int, source_skids_int) if x != y]

    if metric == 'jaccard_sim':
        if weight is not None:
            sim = jaccard_coefficient_weighted(G, weight=weight, ebunch=ebunch)
        else:
            sim = jaccard_coefficient(G, ebunch=ebunch)
    elif metric == 'shared_partners':
        if weight is not None:
            sim = shared_partners_weighted(G, weight=weight, ebunch=ebunch)
        else:
            sim = shared_partners(G, ebunch=ebunch)
    elif metric == 'avg_conv_psd_area':
        sim = avg_conv_psd_area(G, weight=weight, ebunch=ebunch)
    elif metric == 'conv_psd_area_diff':
        sim = conv_psd_area_diff(G, weight=weight, ebunch=ebunch)
    elif metric == 'sp_psd_cos_sim':
        sim = sp_psd_cos_sim(G, weight=weight, ebunch=ebunch)
    else:
        print('Error: metric %s not recognized, reverting to default, jaccard_sim' % metric)
        if weight is not None:
            sim = jaccard_coefficient_weighted(G, weight=weight, ebunch=ebunch)
        else:
            sim = jaccard_coefficient(G, ebunch=ebunch)
    sim_df = pd.DataFrame(sim)
    sim_df = sim_df.rename(columns={0: "source", 1: "target", 2: "weight"})
    sim_G = nx.from_pandas_edgelist(sim_df, edge_attr='weight')
    sim_sq = nx.to_pandas_adjacency(sim_G, nodelist = np.array(source_skids,dtype=int))
    sim_sq_cond = dist.squareform(sim_sq,checks=False)
    return sim_sq, sim_sq_cond

def shared_partners(G, ebunch=None):
    def predict(u, v):
        return len(list(nx.common_neighbors(G, u, v)))
    return _apply_prediction(G, predict, ebunch)

def shared_partners_weighted(G, ebunch=None, weight='count'):
    def predict(u, v):
        other_nodes = set(G.nodes)
        other_nodes.remove(u)
        other_nodes.remove(v)
        u_edges = [G[u][i][weight] if i in G[u] else 0 for i in other_nodes]
        v_edges = [G[v][i][weight] if i in G[v] else 0 for i in other_nodes]

        numer = 0
        denom = 0

        for idx,node in enumerate(other_nodes):
            numer+= np.minimum(u_edges[idx], v_edges[idx])
            denom+= np.maximum(u_edges[idx], v_edges[idx])

        if denom == 0:
            return 0
        else:
            return numer
    return _apply_prediction(G, predict, ebunch)
def avg_conv_psd_area(G, ebunch=None, weight='avg_psd_area'):
    def predict(u, v):
        other_nodes = set(G.nodes)
        other_nodes.remove(u)
        other_nodes.remove(v)
        u_edges = [G[u][i][weight] if i in G[u] else 0 for i in other_nodes]
        v_edges = [G[v][i][weight] if i in G[v] else 0 for i in other_nodes]

        avg_psd_sum = 0
        shared_partner_count = 0

        for idx,node in enumerate(other_nodes):
            if np.minimum(u_edges[idx], v_edges[idx]) > 0:
                 avg_psd_sum += np.average([u_edges[idx], v_edges[idx]])
                 shared_partner_count += 1
        if shared_partner_count == 0:
            return np.nan
        else:
            return avg_psd_sum/shared_partner_count
    return _apply_prediction(G, predict, ebunch)

def conv_psd_area_diff(G, ebunch=None, weight='avg_psd_area'):
    def predict(u, v):
        other_nodes = set(G.nodes)
        other_nodes.remove(u)
        other_nodes.remove(v)
        u_edges = [G[u][i][weight] if i in G[u] else 0 for i in other_nodes]
        v_edges = [G[v][i][weight] if i in G[v] else 0 for i in other_nodes]

        avg_psd_diff = 0
        shared_partner_count = 0

        for idx,node in enumerate(other_nodes):
            if np.minimum(u_edges[idx], v_edges[idx]) > 0:
                 avg_psd_diff += np.absolute(u_edges[idx] - v_edges[idx])
                 shared_partner_count += 1
        if shared_partner_count == 0:
            return np.nan
        else:
            return avg_psd_diff/shared_partner_count
    return _apply_prediction(G, predict, ebunch)

def sp_psd_cos_sim(G, ebunch=None, weight='avg_psd_area'):
    import numpy.ma as ma
    import scipy.spatial.distance as dist
    avg_psd = 179714 #ATK hack for now, need to do more elegantly
    def predict(u, v):
        other_nodes = set(G.nodes)
        other_nodes.remove(u)
        other_nodes.remove(v)
        u_edges = [G[u][i][weight] if i in G[u] else np.nan for i in other_nodes]
        v_edges = [G[v][i][weight] if i in G[v] else np.nan for i in other_nodes]
        a = ma.masked_invalid(u_edges)
        b = ma.masked_invalid(v_edges)
        conv_mask = np.logical_and(~a.mask, ~b.mask)
        a_conv = np.array(u_edges)[conv_mask] - avg_psd
        b_conv = np.array(v_edges)[conv_mask] - avg_psd
        if sum(conv_mask) > 1:
            #print((a_conv,b_conv))
            sim = 1 - dist.euclidean(a_conv,b_conv)
            return sim
        else:
            return np.nan
    return _apply_prediction(G, predict, ebunch)

def jaccard_coefficient_weighted(G, ebunch=None, weight='count'):
    def predict(u, v):
        other_nodes = set(G.nodes)
        other_nodes.remove(u)
        other_nodes.remove(v)
        u_edges = [G[u][i][weight] if i in G[u] else 0 for i in other_nodes]
        v_edges = [G[v][i][weight] if i in G[v] else 0 for i in other_nodes]

        numer = 0
        denom = 0

        for idx,node in enumerate(other_nodes):
            numer+= np.minimum(u_edges[idx], v_edges[idx])
            denom+= np.maximum(u_edges[idx], v_edges[idx])

        if denom == 0:
            return 0
        else:
            return numer / denom
    return _apply_prediction(G, predict, ebunch)

def jaccard_coefficient(G, ebunch=None):
    def predict(u, v):
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / union_size
    return _apply_prediction(G, predict, ebunch)

def _apply_prediction(G, func, ebunch=None):
    """Applies the given function to each edge in the specified iterable
    of edges.

    `G` is an instance of :class:`networkx.Graph`.

    `func` is a function on two inputs, each of which is a node in the
    graph. The function can return anything, but it should return a
    value representing a prediction of the likelihood of a "link"
    joining the two nodes.

    `ebunch` is an iterable of pairs of nodes. If not specified, all
    non-edges in the graph `G` will be used.

    """
    if ebunch is None:
        ebunch = nx.non_edges(G)
    return ((u, v, func(u, v)) for u, v in ebunch)

def get_synapses_between(sources_df, targets_df):
    def calc_syn_approx_D(connector_id):
        from scipy.spatial.distance import euclidean
        try:
            cn_details = pymaid.get_connector_details(connector_id)
            pre_s_node = cn_details.presynaptic_to_node
            post_s_node = cn_details.postsynaptic_to_node
            pre_s_node_loc = pymaid.get_node_location(pre_s_node)
            post_s_node_loc = pymaid.get_node_location(post_s_node[0][0])
            syn_D = euclidean(pre_s_node_loc[['x','y','z']],post_s_node_loc[['x','y','z']])
            return syn_D
        except:
            return np.nan
    def classify_syn_types(synapse_df, sources_df, targets_df):
        def get_types(skel_id, lookup_df):
            try:
                return lookup_df[lookup_df['skeleton_id'] == skel_id]['type'].values[0]
            except:
                print('ERROR: skel_id %i has no neuron type' % skel_id)
                return 'unknown'
        def get_layers(skel_id, lookup_df):
            try:
                return lookup_df[lookup_df['skeleton_id'] == skel_id]['layer'].values[0]
            except:
                print('ERROR: skel_id %i has no layer info' % skel_id)
                return 'unknown'
        def get_soma_loc(skel_id, lookup_df):
            try:
                return lookup_df[lookup_df['skeleton_id'] == skel_id]['soma_loc'].values[0]
            except:
                print('ERROR: skel_id %i has no soma location info' % skel_id)
                return (np.nan, np.nan, np.nan)
        synapse_df['source_type'] = synapse_df.apply (lambda row:
            get_types(row['source'], sources_df), axis=1)
        synapse_df['target_type'] = synapse_df.apply (lambda row:
            get_types(row['target'], targets_df), axis=1)
        synapse_df['target_layer'] = synapse_df.apply (lambda row:
            get_layers(row['target'], targets_df), axis=1)
        synapse_df['target_loc'] = synapse_df.apply (lambda row:
            get_soma_loc(row['target'], targets_df), axis=1)
        return synapse_df
    mySources = pymaid.get_neuron(sources_df.skeleton_id.astype(int))
    myTargets = pymaid.get_neuron(targets_df.skeleton_id.astype(int))
    connectors = pymaid.get_connectors_between(mySources,myTargets, directional=True)
    synapse_df = connectors[['source_neuron','target_neuron','connector_id','connector_loc']]
    if len(synapse_df) > 0:
        # units are um^2
        synapse_df['psd_area'] = synapse_df.apply (lambda row:
            3.14159*(calc_syn_approx_D(row['connector_id'])/2)**2/1e6, axis=1)
        synapse_df["count"] = 1
        synapse_df = synapse_df.rename(columns={"source_neuron": "source", "target_neuron": "target"})
        synapse_df = classify_syn_types(synapse_df, sources_df, targets_df)
    else:
        print('no synapses')
    return synapse_df

def cn_df_from_syn_df(synapse_df, neuron_df, selectIdx='RL_selectIdx', selectivity='RL_selectivity'):
    def add_source_type(row, syn_df):
        return syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)].iloc[0]['source_type']
    def add_target_type(row, syn_df):
        return syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)].iloc[0]['target_type']
    def add_source_select(row, neuron_df):
        try:
            return neuron_df[(neuron_df.skeleton_id==row.source)][selectIdx].values[0]
        except:
            #print('skelID %i has no selectivity' % row.source)
            return np.nan
    def add_target_select(row, neuron_df):
        try:
            return neuron_df[(neuron_df.skeleton_id==row.target)][selectIdx].values[0]
        except:
            #print('skelID %i has no selectivity' % row.target)
            return np.nan
    def add_source_select_class(row, neuron_df):
        try:
            return neuron_df[(neuron_df.skeleton_id==row.source)][selectivity].values[0]
        except:
            #print('skelID %i has no selectivity class' % row.source)
            return np.nan
    def add_target_select_class(row, neuron_df):
        try:
            return neuron_df[(neuron_df.skeleton_id==row.target)][selectivity].values[0]
        except:
            #print('skelID %i has not selectivity class' % row.target)
            return np.nan
    def add_RL_pair_types(row, neuron_df):
        try:
            RL_1 = row.source_selectivity
            RL_2 = row.target_selectivity
            if RL_1 != 'Non' and RL_1 != 'Mixed' and RL_2 != 'Non' and RL_2 !='Mixed':
                if RL_1 == RL_2:
                    RL_pair_type = 'same'
                else:
                    RL_pair_type = 'opposite'
            #elif RL_1 == 'Non' and RL_2 == 'Non':
            #    RL_pair_types.iloc[idx_1,idx_2] = 'same'
            else:
                RL_pair_type = 'NA'

            return RL_pair_type
        except:
            return np.nan
    def add_RL_diff(row):
        return np.abs(row['source_select']-row['target_select'])
    def add_RL_sim(row):
        return 1-np.abs(row['source_select']-row['target_select'])
    def add_psd_areas(row, syn_df):
        syns = syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)]
        return np.round(syns.psd_area.values)
    def add_avg_psd_area(row, syn_df):
        syns = syn_df[(syn_df.source==row.source) & (syn_df.target==row.target)]
        return np.round(np.average(syns.psd_area.values))
    def add_connector_ids(row, syn_df):
        return syn_df[syn_df.source==row.source][syn_df.target==row.target].connector_id.values
    if len(synapse_df) > 0:
        DiG =DiGraph_from_syn(synapse_df)
        cn_df = nx.to_pandas_edgelist(DiG)
        cn_df['source_type'] = cn_df.apply (lambda row: add_source_type(row, synapse_df), axis=1)
        cn_df['target_type'] = cn_df.apply (lambda row: add_target_type(row, synapse_df), axis=1)
        cn_df['source_select'] = cn_df.apply (lambda row: add_source_select(row, neuron_df), axis=1)
        cn_df['target_select'] = cn_df.apply (lambda row: add_target_select(row, neuron_df), axis=1)
        cn_df['source_selectivity'] = cn_df.apply (lambda row: add_source_select_class(row, neuron_df), axis=1)
        cn_df['target_selectivity'] = cn_df.apply (lambda row: add_target_select_class(row, neuron_df), axis=1)
        cn_df['selectivity'] = cn_df.apply (lambda row: add_RL_pair_types(row, neuron_df), axis=1)
        cn_df['RL_diff'] =  cn_df.apply (lambda row: add_RL_diff(row), axis=1)
        cn_df['RL_sim'] =  cn_df.apply (lambda row: add_RL_sim(row), axis=1)
        cn_df['psd_areas'] = cn_df.apply (lambda row: add_psd_areas(row, synapse_df), axis=1)
        cn_df['avg_psd_area'] = cn_df.apply (lambda row: add_avg_psd_area(row, synapse_df), axis=1)
        cn_df['cn_ids'] = cn_df.apply (lambda row: add_connector_ids(row, synapse_df), axis=1)
    return cn_df

def DiGraph_from_syn(synapse_df):
    try:
        DiG = nx.from_pandas_edgelist(synapse_df.groupby(['source', 'target'], as_index=False).agg('sum'),
            edge_attr=['avg_psd_area','psd_area','count'],create_using=nx.DiGraph)
    except:
        DiG = nx.from_pandas_edgelist(synapse_df.groupby(['source', 'target'], as_index=False).agg('sum'),
            edge_attr=['psd_area','count'],create_using=nx.DiGraph)
    return DiG

def plot_DiGraph(DiG, weight='psd_area'):
    #fig, ax = plt.subplots(figsize=(20,20))
    edge_widths = np.array([i[weight] for i in dict(DiG.edges).values()])
    # normalize edge widths
    edge_widths = edge_widths/np.mean(edge_widths)*5
    pos = nx.spring_layout(DiG,weight=weight)
    nx.draw_networkx_nodes(DiG, pos, labels=True)
    nx.draw_networkx_edges(DiG, pos, width=edge_widths)

def get_shared_partners(source_skids, target_skids):
    adj_mat = calc_adj_mat(source_skids, targets = target_skids)
    shared_connections_mat = np.zeros([len(source_skids), len(source_skids)])
    for idx1,source1 in enumerate(source_skids):
        for idx2,source2 in enumerate(source_skids):
            nz1 = np.nonzero(adj_mat.loc[int(source1)].values)
            nz2 = np.nonzero(adj_mat.loc[int(source2)].values)
            shared_connections_mat[idx1,idx2] = len(np.intersect1d(nz1,nz2))
        shared_connections_mat[idx1,idx1] = 0
    return shared_connections_mat, dist.squareform(shared_connections_mat,checks=False)

def shared_partners_from_nx(G, source_skids):
    j = nx.cn_soundarajan_hopcroft(G)
    j_df = pd.DataFrame(j)
    j_df = j_df.rename(columns={0: "source", 1: "target", 2: "weight"})
    j_G = nx.from_pandas_edgelist(j_df, edge_attr='weight')
    j_sq = nx.to_pandas_adjacency(j_G, nodelist = np.array(source_skids,dtype=int))
    return j_sq

def get_jaccard_bootstrap(source_skids, target_skids, num_samp, lower, upper):
    j_sim_bootstrap = np.zeros((len(source_skids),len(source_skids),num_samp))
    synapse_df = get_synapses_between(source_skids, target_skids)
    G = nx.from_pandas_edgelist(synapse_df)
    j_sim = jaccard_sim_from_nx(G, source_skids)
    for i in np.arange(num_samp):
        synapse_bs = bootstrap_synapses(synapse_df)
        G_bs = nx.from_pandas_edgelist(synapse_bs, edge_attr='weight')
        plt.subplot(num_samp,1,i+1)
        nx.draw_shell(G_bs, with_labels = True, edge_cmap = 'jet', vmax = 3)
        j_sq_bs = jaccard_sim_from_nx(G_bs, source_skids)
        j_sim_bootstrap[:,:,i] = j_sq_bs
    j_sim_upper = np.percentile(j_sim_bootstrap, upper, axis=2)
    j_sim_lower = np.percentile(j_sim_bootstrap, lower, axis=2)

    return j_sim, j_sim_lower, j_sim_upper

def calc_act_metric_diff(myNeuronsDf, RL_select, mySessions):

    diff_mat_combined = np.empty([len(myNeuronsDf), len(myNeuronsDf), len(mySessions)])
    sessions_overlap = np.zeros([len(myNeuronsDf), len(myNeuronsDf)])
    diff_mat_combined[:] = np.nan

    for idx_1,id_1 in enumerate(myNeuronsDf.matched_cell_ID):
        neuron_1 = myNeuronsDf[myNeuronsDf.matched_cell_ID == id_1]
        for idx_2, id_2 in enumerate(myNeuronsDf.matched_cell_ID):
            neuron_2 = myNeuronsDf[myNeuronsDf.matched_cell_ID == id_2]
            sessions_1_bool = np.array([x>-1 for x in neuron_1.sessions_ROI])
            sessions_2_bool = np.array([x>-1 for x in neuron_2.sessions_ROI])
            match_bool = np.logical_and(sessions_1_bool, sessions_2_bool)
            for session_idx,match in enumerate(match_bool[0]):
                if match:
                    roi_1 = int(neuron_1.sessions_ROI.values[0][session_idx]) # This hacky indexing needs to be fixed when i have time
                    roi_2 = int(neuron_2.sessions_ROI.values[0][session_idx])
                    if idx_1 != idx_2:
                        diff_mat_combined[idx_1, idx_2, session_idx] = np.absolute(RL_select[mySessions[session_idx]][0,roi_1] - RL_select[mySessions[session_idx]][0,roi_2])
                        sessions_overlap[idx_1,idx_2] += 1;
    diff_mat_average = np.nanmean(diff_mat_combined,2)
    return diff_mat_average, dist.squareform(diff_mat_average,checks=False),sessions_overlap

def plot_my_corr(myCorr,labels,title,cmap='hot',cbar=False):
    #sns.set(font_scale = 2)
    g = sns.heatmap(myCorr, square=True,cmap=cmap,cbar=cbar)
    g.set_xticklabels(labels)
    g.set_xticklabels(labels, rotation=90)
    g.set_yticklabels(labels, rotation=0)
    g.set_title(title)

def calc_morphology_overlap(myNeuronsDf, labels, max_dist):

    soma_locs = np.array([i for i in myNeuronsDf.soma_loc.values])#[:,:,0]
    soma_dists = dist.pdist(soma_locs)/1000
    soma_dists_xy = dist.pdist(soma_locs[:,[0,1]])/1000
    soma_dists_xz = dist.pdist(soma_locs[:,[0,2]])/1000
    soma_dists_yz = dist.pdist(soma_locs[:,[1,2]])/1000
    soma_dists_x = dist.pdist(soma_locs[:,[0]])/1000

    myNeurons = pymaid.get_neurons(myNeuronsDf[:].skeleton_id)
    (myList, trunkList, collatList) = (list(), list(), list())

    for idx,myNeuron in enumerate(myNeurons):
        myLabels = labels[ labels.skeleton_id == int(myNeuron.skeleton_id) ]
        (myAxon, myAxonTrunk, myAxonCollaterals, myMyelination) = get_axon_components(myNeuron, myLabels);
        myList.append(myAxon)
        trunkList.append(myAxonTrunk)
        collatList.append(myAxonCollaterals)
    myAxons = pymaid.core.CatmaidNeuronList(myList)
    myAxonTrunks = pymaid.core.CatmaidNeuronList(trunkList)
    myAxonCollats = pymaid.core.CatmaidNeuronList(collatList)

    if len(myAxons[myAxons.n_nodes < 2]) > 0:
        print('Some nodes have no axon')
        print(myAxons[myAxons.n_nodes < 2])
        return (soma_dists, None, None, None)
    axon_overlap = dist.squareform(navis.cable_overlap(myAxons, myAxons, dist=max_dist, method='min'), checks=False)
    if len(myAxons[myAxonTrunks.n_nodes < 2]) > 0:
        print('Some nodes have no trunk (probably no thickness on skeleton)')
        print(myAxons[myAxonTrunks.n_nodes < 2])
        return (soma_dists, axon_overlap, None, None)
    trunk_overlap = dist.squareform(navis.cable_overlap(myAxonTrunks, myAxonTrunks, dist=max_dist, method='min'), checks=False)
    if len(myAxons[myAxonCollats.n_nodes < 2]) > 0:
        print('Some nodes have no trunk (probably no thickness on skeleton)')
        print(myAxons[myAxonCollats.n_nodes < 2])
        return (soma_dists, axon_overlap, trunk_overlap, None)
    collat_overlap = dist.squareform(navis.cable_overlap(myAxonCollats, myAxonCollats, dist=max_dist, method='min'), checks=False)
    return (soma_dists, axon_overlap, trunk_overlap, collat_overlap)

def calc_RL_pair_types(myNeuronsDf):
    RL_pair_types = pd.DataFrame(index=myNeuronsDf.matched_cell_ID, columns=myNeuronsDf.matched_cell_ID)

    np.empty([len(myNeuronsDf), len(myNeuronsDf)],dtype=np.string_)
    for idx_1,RL_1 in enumerate(myNeuronsDf.RL_selectivity):
        for idx_2, RL_2 in enumerate(myNeuronsDf.RL_selectivity):
            if RL_1 != 'Non' and RL_2 != 'Non':
                if RL_1 == RL_2:
                    RL_pair_types.iloc[idx_1,idx_2] = 'same'
                else:
                    RL_pair_types.iloc[idx_1,idx_2] = 'opposite'
            #elif RL_1 == 'Non' and RL_2 == 'Non':
            #    RL_pair_types.iloc[idx_1,idx_2] = 'same'
            else:
                RL_pair_types.iloc[idx_1,idx_2] = 'NA'
    return RL_pair_types, dist.squareform(RL_pair_types,checks=False)

def calc_select_pair_types(myNeuronsDf, select_metric = 'RL_selectivity'):
    RL_pair_types = pd.DataFrame(index=myNeuronsDf.matched_cell_ID, columns=myNeuronsDf.matched_cell_ID)

    np.empty([len(myNeuronsDf), len(myNeuronsDf)],dtype=np.string_)
    for idx_1,RL_1 in enumerate(myNeuronsDf.RL_selectivity):
        for idx_2, RL_2 in enumerate(myNeuronsDf.RL_selectivity):
            if RL_1 != 'Non' and RL_2 != 'Non':
                if RL_1 == RL_2:
                    RL_pair_types.iloc[idx_1,idx_2] = 'same'
                else:
                    RL_pair_types.iloc[idx_1,idx_2] = 'opposite'
            #elif RL_1 == 'Non' and RL_2 == 'Non':
            #    RL_pair_types.iloc[idx_1,idx_2] = 'same'
            else:
                RL_pair_types.iloc[idx_1,idx_2] = 'NA'
    return RL_pair_types, dist.squareform(RL_pair_types,checks=False)

def calc_trialSNR_pair_types(myNeuronsDf):
    snr_pair_types = pd.DataFrame(index=myNeuronsDf.matched_cell_ID, columns=myNeuronsDf.matched_cell_ID)
    thresh = .1
    np.empty([len(myNeuronsDf), len(myNeuronsDf)],dtype=np.string_)
    for idx_1,snr_1 in enumerate(myNeuronsDf.trial_snr_max):
        for idx_2, snr_2 in enumerate(myNeuronsDf.trial_snr_max):
            if snr_1 < thresh and snr_2 < thresh:
                snr_pair_types.iloc[idx_1,idx_2] = 'both low'
            elif snr_1 > thresh and snr_2 > thresh:
                snr_pair_types.iloc[idx_1,idx_2] = 'both high'
            else:
                snr_pair_types.iloc[idx_1,idx_2] = 'mixed'

    return snr_pair_types, dist.squareform(snr_pair_types,checks=False)

def scatter_and_fit(X, y, xlabel=None, ylabel=None, plot=True, fit_intercept = True):
    from sklearn.linear_model import LinearRegression
    X_ = X.reshape(-1, 1)
    y_ = y.reshape(-1, 1)
    reg = LinearRegression(fit_intercept=fit_intercept).fit(X_, y_)
    y_pred = reg.predict(X_)

    if plot == True:
        sns.scatterplot(X, y)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.plot(X,y_pred)
        plt.title('R2 = %0.2f' % reg.score(X_,y_))
    return reg, reg.score(X_,y_)

#def query_MI_selectivity(myNeuronDf, myCellID, mySessions, choiceMI_max, choiceMI_prctile, 
#    choiceMI_pref, select_metric = 'MI', sig_thresh = 0.99):
# ATK 220221 remove choice_prctile for now to save time
def query_MI_selectivity(myNeuronDf, myCellID, mySessions, choiceMI_max, 
    choiceMI_pref, select_metric = 'MI', sig_thresh = 0.95, column = 0):
    # Columns used for epoch-wise, where 5 columns are different epochs
    # For default, col = 0 is the only value
    select_metric = '_'+select_metric
    myMIselectIdx = np.nan
    myMIselectIdxSum = 0
    num_active_sessions = 0
    sessions_ROI = myNeuronDf['sessions_ROI']
    sessions_select_idx = np.empty(len(mySessions))
    sessions_select_idx[:] = np.nan
    LNR_select = np.array([0,0,0])

    for session_idx,session in enumerate(mySessions):
        if sessions_ROI[session_idx]>-1:
            num_active_sessions += 1
            myChoiceMI_pref = choiceMI_pref[session][column,int(sessions_ROI[session_idx])]
            if myChoiceMI_pref >= 0:
                myMIselectIdx = choiceMI_max[session][column,int(sessions_ROI[session_idx])]
                #if choiceMI_prctile[session][0,int(sessions_ROI[session_idx])]>sig_thresh:
                #    LNR_select += [0,0,1]
                #else:
                #    LNR_select += [0,1,0]
            else: # myRLselectIdx < 0
                myMIselectIdx = -choiceMI_max[session][column,int(sessions_ROI[session_idx])]
                #if choiceMI_prctile[session][0,int(sessions_ROI[session_idx])]>sig_thresh:
                #    LNR_select += [1,0,0]     
                #else:
                #    LNR_select += [0,1,0]   
            myMIselectIdxSum += myMIselectIdx
            sessions_select_idx[session_idx] = myMIselectIdx
    if num_active_sessions > 0:
        myMIselectIdx = myMIselectIdxSum/num_active_sessions
        myNeuronDf['select_idx'+select_metric] = myMIselectIdx
        myNeuronDf['select_idx'+select_metric+'_abs'] = np.absolute(myMIselectIdx)
        #myNeuronDf['LNR_select'+select_metric] = LNR_select
        myNeuronDf['sessions_select_idx'+select_metric] = sessions_select_idx
        myNeuronDf['select_idx'+select_metric+'_std'] = np.nanstd(sessions_select_idx)
        myNeuronDf['select_idx'+select_metric+'_stdmean'] = np.nanstd(sessions_select_idx)/np.sqrt(num_active_sessions)
        
        LNR_max = [i for i, j in enumerate(LNR_select) if j == max(LNR_select)]
        # 220114 new scheme for calculating overall selectivity - more lenient
        if LNR_select[0] > 0 or LNR_select[2] > 0:
            if LNR_select[0] > LNR_select[2]:
                myNeuronDf['selectivity'+select_metric] = 'Left'
            elif LNR_select[0] < LNR_select[2]:
                myNeuronDf['selectivity'+select_metric] = 'Right'
            else:
                myNeuronDf['selectivity'+select_metric] = 'Mixed'
        else:
            myNeuronDf['selectivity'+select_metric] = 'Non'    
            
        ''' old scheme - more stringent
        if len(LNR_max) > 1:
            if LNR_select[0] > LNR_select[2]:
                myNeuronDf['selectivity'+select_metric] = 'Left'
            elif LNR_select[0] < LNR_select[2]:
                myNeuronDf['selectivity'+select_metric] = 'Right'
            else:
                myNeuronDf['selectivity'+select_metric] = 'Mixed'
        elif LNR_max[0] == 0:
            myNeuronDf['selectivity'+select_metric] = 'Left'
        elif LNR_max[0] == 1:
            myNeuronDf['selectivity'+select_metric] = 'Non'
        elif LNR_max[0] == 2:
            myNeuronDf['selectivity'+select_metric] = 'Right'
        '''

def query_MI_selectivity_t(myNeuronDf, myCellID, mySessions, select_idx_t,
    select_metric = 'MI_t', filter=False, choiceMI_sig_t=None):
    # ATK 230202 adding t-wise select idx.
    # ATK 230606 adding filter based on significance vs shuffle

    # For default, col = 0 is the only value
    select_metric = '_'+select_metric
    myMIselectIdx = np.empty(np.size(select_idx_t[mySessions[0]],0))
    myMIselectIdx[:] = np.nan
    myMIselectIdxSum = np.empty(np.size(select_idx_t[mySessions[0]],0))
    myMIselectIdxSum[:] = 0
    num_active_sessions = 0
    sessions_ROI = myNeuronDf['sessions_ROI']
    sessions_select_idx = np.empty((len(mySessions), np.size(select_idx_t[mySessions[0]],0)))
    sessions_select_idx[:] = np.nan
    LNR_select = np.array([0,0,0])

    for session_idx,session in enumerate(mySessions):
        if sessions_ROI[session_idx]>-1:
            num_active_sessions += 1
            myMIselectIdx = select_idx_t[session][:,int(sessions_ROI[session_idx])]
            if filter and choiceMI_sig_t is not None:
                myMI_sig_t = choiceMI_sig_t[session][:,int(sessions_ROI[session_idx])]
                myMIselectIdx = np.multiply(myMIselectIdx, myMI_sig_t[13:]) # filter non-sig MI to 0
                # select_idx already excludes ITIbefore
            myMIselectIdxSum += myMIselectIdx
            sessions_select_idx[session_idx,:] = myMIselectIdx
    if num_active_sessions > 0:
        myMIselectIdx = myMIselectIdxSum/num_active_sessions
        myNeuronDf['select_idx'+select_metric] = myMIselectIdx
        myNeuronDf['sessions_select_idx'+select_metric] = sessions_select_idx

def query_MI_selectivity_t_prctile(myNeuronDf, myCellID, mySessions, select_idx_t,
    select_metric = 'MI_t', filter=True, choiceMI_prctile=None, thresh = 0.95):
    # ATK 230202 adding t-wise select idx.
    # ATK 230606 adding filter based on significance vs shuffle

    # For default, col = 0 is the only value
    select_metric = '_'+select_metric
    myMIselectIdx = np.empty(np.size(select_idx_t[mySessions[0]],0))
    myMIselectIdx[:] = np.nan
    myMIselectIdxSum = np.empty(np.size(select_idx_t[mySessions[0]],0))
    myMIselectIdxSum[:] = 0
    num_active_sessions = 0
    sessions_ROI = myNeuronDf['sessions_ROI']
    sessions_select_idx = np.empty((len(mySessions), np.size(select_idx_t[mySessions[0]],0)))
    sessions_select_idx[:] = np.nan
    LNR_select = np.array([0,0,0])

    for session_idx,session in enumerate(mySessions):
        if sessions_ROI[session_idx]>-1:
            num_active_sessions += 1
            myMIselectIdx = select_idx_t[session][:,int(sessions_ROI[session_idx])]
            if filter and choiceMI_prctile is not None:
                myMI_prctile = choiceMI_prctile[session][:,int(sessions_ROI[session_idx])]
                myMI_sig_t = myMI_prctile > thresh
                myMIselectIdx = np.multiply(myMIselectIdx, myMI_sig_t[13:]) # filter non-sig MI to 0
                # select_idx already excludes ITIbefore
            myMIselectIdxSum += myMIselectIdx
            sessions_select_idx[session_idx,:] = myMIselectIdx
    if num_active_sessions > 0:
        myMIselectIdx = myMIselectIdxSum/num_active_sessions
        myNeuronDf['select_idx'+select_metric] = myMIselectIdx
        myNeuronDf['sessions_select_idx'+select_metric] = sessions_select_idx


def query_selectivity(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,select_idx,select_idx_blocks,
         selectivity, selectivity_blocks, select_metric = ''):
    # select_metric used as label in dataframe columns
    select_metric = '_'+select_metric

    myRLselectIdx = None
    myRLselectIdxSum = 0
    myRLselectIdx_blocks = np.empty(5)
    myRLselectIdxSum_blocks = np.zeros(5)
    num_active_sessions = 0
    LNR_select = np.array([0,0,0])
    LNR_select_blocks = dict()
    
    blocks = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
    for (i, block) in enumerate(blocks):
        LNR_select_blocks[block] = np.array([0,0,0])

    sessions_ROI = myNeuronDf['sessions_ROI']
    for session_idx,session in enumerate(mySessions):
        if sessions_ROI[session_idx]>-1:
            myRLselectIdxSum += select_idx[session][0,int(sessions_ROI[session_idx])]
            myRLselectIdxSum_blocks += select_idx_blocks[session][:,int(sessions_ROI[session_idx])]
            num_active_sessions += 1
            if selectivity[session][0,int(sessions_ROI[session_idx])] == 2:
                LNR_select += [1,0,0]
            elif selectivity[session][0,int(sessions_ROI[session_idx])] == 3:
                LNR_select += [0,0,1]
            elif selectivity[session][0,int(sessions_ROI[session_idx])] == 0:
                LNR_select += [0,1,0]
            # check indent below?
            for (i, block) in enumerate(blocks):
                if selectivity_blocks[session][i,int(sessions_ROI[session_idx])] == 2:
                    LNR_select_blocks[block] += [1,0,0]
                elif selectivity_blocks[session][i,int(sessions_ROI[session_idx])] == 3:
                    LNR_select_blocks[block] += [0,0,1]
                elif selectivity_blocks[session][i,int(sessions_ROI[session_idx])] == 0:
                    LNR_select_blocks[block] += [0,1,0]
    if num_active_sessions > 0:
        myRLselectIdx = myRLselectIdxSum/num_active_sessions
        myRLselectIdx_blocks =  myRLselectIdxSum_blocks/num_active_sessions
    else:
        myRLselectIdx = np.nan
        myRLselectIdx_blocks.fill(np.nan)

    # rescale to [-1 1]
    if select_metric == '_ROC':
        myRLselectIdx = 2*(myRLselectIdx-.5)
        myRLselectIdx_blocks = 2*(myRLselectIdx_blocks-.5)
    myNeuronDf['select_idx'+select_metric] = myRLselectIdx
    myNeuronDf['select_idx'+select_metric+'_blocks'] = myRLselectIdx_blocks
    myNeuronDf['LNR_select'+select_metric] = LNR_select
    LNR_max = [i for i, j in enumerate(LNR_select) if j == max(LNR_select)]
    # 220114 new scheme for calculating overall selectivity - more lenient
    if LNR_select[0] > 0 or LNR_select[2] > 0:
        if LNR_select[0] > LNR_select[2]:
            myNeuronDf['selectivity'+select_metric] = 'Left'
        elif LNR_select[0] < LNR_select[2]:
            myNeuronDf['selectivity'+select_metric] = 'Right'
        else:
            myNeuronDf['selectivity'+select_metric] = 'Mixed'
    else:
        myNeuronDf['selectivity'+select_metric] = 'Non'
    '''
    # old LNR scheme
    if len(LNR_max) > 1:
        if LNR_select[0] > LNR_select[2]:
            myNeuronDf['selectivity'+select_metric] = 'Left'
        elif LNR_select[0] < LNR_select[2]:
            myNeuronDf['selectivity'+select_metric] = 'Right'
        else:
            myNeuronDf['selectivity'+select_metric] = 'Mixed'
    elif LNR_max[0] == 0:
        myNeuronDf['selectivity'+select_metric] = 'Left'
    elif LNR_max[0] == 1:
        myNeuronDf['selectivity'+select_metric] = 'Non'
    elif LNR_max[0] == 2:
        myNeuronDf['selectivity'+select_metric] = 'Right'
    '''
    '''
    for (i, block) in enumerate(blocks):
        # add blockwise select idx
        myNeuronDf['select_idx'+select_metric+'_'+block] = myRLselectIdx_blocks[i]

        # add blockwise selectivity class
        LNR_max = [i for i, j in enumerate(LNR_select_blocks[block]) if j == max(LNR_select_blocks[block])]
        if len(LNR_max) > 1:
            if LNR_select_blocks[block][0] > LNR_select_blocks[block][2]:
                myNeuronDf['selectivity'+select_metric+'_'+block] = 'Left'
            elif LNR_select_blocks[block][0] < LNR_select_blocks[block][2]:
                myNeuronDf['selectivity'+select_metric+'_'+block] = 'Right'
            else:
                myNeuronDf['selectivity'+select_metric+'_'+block] = 'Mixed'
        elif LNR_max[0] == 0:
            myNeuronDf['selectivity'+select_metric+'_'+block] = 'Left'
        elif LNR_max[0] == 1:
            myNeuronDf['selectivity'+select_metric+'_'+block] = 'Non'
        elif LNR_max[0] == 2:
            myNeuronDf['selectivity'+select_metric+'_'+block] = 'Right'
    '''
def query_t_peak(myNeuronDf,myCellID,myNeuron,myLabels,mySessions,t_peak, label='t_peak'):
    my_t_peak = None
    my_t_peak_sum = 0
    num_active_sessions = 0
    sessions_ROI = myNeuronDf['sessions_ROI']
    for session_idx,session in enumerate(mySessions):
        if sessions_ROI[session_idx]>-1:
            if label == 'tCOM':
                if myNeuronDf['select_idx_RL']<1:
                    my_t_peak_sum += t_peak[session]['wL_trials'][0,int(sessions_ROI[session_idx])] 
                else:
                    my_t_peak_sum += t_peak[session]['bR_trials'][0,int(sessions_ROI[session_idx])] 
            else: 
                my_t_peak_sum += t_peak[session][0,int(sessions_ROI[session_idx])]
            num_active_sessions += 1
    if num_active_sessions > 0:
        my_t_peak = my_t_peak_sum/num_active_sessions    
    else:
        my_t_peak = np.nan
    myNeuronDf[label] = my_t_peak
    epoch = get_epoch(my_t_peak)
    myNeuronDf[label+'_epoch'] = epoch

def get_epoch(my_t_peak):
    if my_t_peak < 13:
        epoch = 'ITIbefore'
    elif my_t_peak < 27:
        epoch = 'cueEarly'
    elif my_t_peak < 39:
        epoch = 'cueLate'
    elif my_t_peak < 52:
        epoch = 'delay'
    elif my_t_peak < 64:
        epoch = 'turn'
    elif my_t_peak < 77:
        epoch = 'ITI'
    else:
        epoch = None
    return epoch

def calc_outDeg(mySources,syn_df):
    import networkx as nx
    DiG =DiGraph_from_syn(syn_df)
    outDeg_psd_area = pd.DataFrame(DiG.out_degree(DiG,weight='psd_area'),columns=['skel_id','total_psd_area'])
    outDeg_psd_area = outDeg_psd_area[outDeg_psd_area.skel_id.isin(mySources.skeleton_id.values)].set_index('skel_id')
    outDeg_count = pd.DataFrame(DiG.out_degree(DiG,weight='count'),columns=['skel_id','count'])
    outDeg_count = outDeg_count[outDeg_count.skel_id.isin(mySources.skeleton_id.values)].set_index('skel_id')
    #outDeg_count = pd.DataFrame(DiG.out_degree(DiG,weight=None),columns=['skel_id','count'])
    outDeg = pd.concat([outDeg_psd_area, outDeg_count],axis=1)
    outDeg['avg_out_psd_area'] = outDeg.apply (lambda row:
            row['total_psd_area']/row['count'], axis=1)
    #print(outDeg)
    #mySources.set_index('skeleton_id')
    #mySources['avg_out_psd_area'] = outDeg.loc[np.asarray(mySources.skeleton_id.values,dtype=int)]['avg_out_psd_area']
    mySources = pd.concat([mySources, outDeg],axis=1)
    #mySources = mySources.join(outDeg)
    return mySources

def calc_syn_size_dists(syn_df, source_type=None, target_type=None):
    if source_type is not None:
        syn_df = syn_df[syn_df.source_type == source_type]
    if target_type is not None:
        syn_df = syn_df[syn_df.target_type == target_type]
    syn_df['source_select_idx_abs'] = np.abs(syn_df.source_select_idx)
    syn_df_src = syn_df.groupby(by='source').agg(list)
    syn_df_src['avg_psd_area'] = syn_df_src.apply (lambda row: np.mean(row['psd_area']), axis=1)
    syn_df_src['syn_count'] = syn_df_src.apply (lambda row: len(row['psd_area']), axis=1)
    syn_df_src['log_avg_psd_area'] = syn_df_src.apply (lambda row: np.log(np.mean(row['psd_area'])), axis=1)
    syn_df_src['std_psd_area'] = syn_df_src.apply (lambda row: np.std(row['psd_area']), axis=1)
    syn_df_src['stdmean_psd_area'] = syn_df_src.apply (lambda row: np.std(row['psd_area'])/np.sqrt(len(row['psd_area'])), axis=1)

    syn_df_src['avg_log_psd_area'] = syn_df_src.apply (lambda row: np.mean(row['log_psd_area']), axis=1)
    syn_df_src['std_log_psd_area'] = syn_df_src.apply (lambda row: np.std(row['log_psd_area']), axis=1)
    syn_df_src['stdmean_log_psd_area'] = syn_df_src.apply (lambda row: np.std(row['log_psd_area'])/np.sqrt(len(row['log_psd_area'])), axis=1)

    syn_df_src['select_idx'] = syn_df_src.apply (lambda row: (row['source_select_idx'][0]), axis=1)
    syn_df_src['select_idx_std'] = syn_df_src.apply (lambda row: (row['source_select_idx_std'][0]), axis=1)
    syn_df_src['select_idx_stdmean'] = syn_df_src.apply (lambda row: (row['source_select_idx_std'][0]/np.sqrt(row['source_num_active_sessions'][0])), axis=1)
    syn_df_src['selectivity'] = syn_df_src.apply (lambda row: (row['source_selectivity'][0]), axis=1)
    syn_df_src['select_idx_abs'] = np.abs(syn_df_src['select_idx'])
    syn_df_src['trial_snr'] = syn_df_src.apply (lambda row: (row['trial_snr'][0]), axis=1)

    syn_df_src = syn_df_src[syn_df_src.std_psd_area>0] # remove neurons with only one psd and thus no std
    return syn_df_src

def act_corr_old(id_1, id_2, corr_aligned, mySessions, myNeuronsDf, field = None):
    corr_mat_combined = np.empty(len(mySessions))
    sessions_overlap = np.zeros([len(myNeuronsDf), len(myNeuronsDf)])
    corr_mat_combined[:] = np.nan

    neuron_1 = myNeuronsDf[myNeuronsDf.skeleton_id == id_1]
    neuron_2 = myNeuronsDf[myNeuronsDf.skeleton_id == id_2]
    sessions_1_bool = np.array([x>-1 for x in neuron_1.sessions_ROI])
    sessions_2_bool = np.array([x>-1 for x in neuron_2.sessions_ROI])
    match_bool = np.logical_and(sessions_1_bool, sessions_2_bool)
    for session_idx,match in enumerate(match_bool[0]):
        if match:
            roi_1 = int(neuron_1.sessions_ROI.values[0][session_idx]) # This hacky indexing needs to be fixed when i have time
            roi_2 = int(neuron_2.sessions_ROI.values[0][session_idx])
            if id_1 != id_2:
                sessions_overlap += 1
                if field is not None:
                    corr_mat_combined[session_idx] = corr_aligned[mySessions[session_idx]][field][roi_1,roi_2]
                else:
                    corr_mat_combined[session_idx] = corr_aligned[mySessions[session_idx]][roi_1,roi_2]
    if sum(~np.isnan(corr_mat_combined))>0:
        corr_mat_average = np.nanmean(corr_mat_combined)
    else:
        corr_mat_average = np.nan
    return corr_mat_average

def act_corr(id_1, id_2, corr_aligned, mySessions, myNeuronsDf, field = None):
    corr_mat_combined = np.empty(len(mySessions))
    sessions_overlap = np.zeros([len(myNeuronsDf), len(myNeuronsDf)])
    corr_mat_combined[:] = np.nan

    neuron_1 = myNeuronsDf[myNeuronsDf.skeleton_id == id_1]
    neuron_2 = myNeuronsDf[myNeuronsDf.skeleton_id == id_2]
    sessions_1_bool = np.array([x>-1 for x in neuron_1.sessions_ROI][0])
    sessions_2_bool = np.array([x>-1 for x in neuron_2.sessions_ROI][0])
    match_bool = np.logical_and(sessions_1_bool, sessions_2_bool)
    for session_idx,match in enumerate(match_bool):
        if match:
            roi_1 = int(neuron_1.sessions_ROI.values[0][session_idx]) 
            roi_2 = int(neuron_2.sessions_ROI.values[0][session_idx])
            if id_1 != id_2:
                sessions_overlap += 1
                if field is not None:
                    corr_mat_combined[session_idx] = corr_aligned[mySessions[session_idx]][field][roi_1,roi_2]
                else:
                    corr_mat_combined[session_idx] = corr_aligned[mySessions[session_idx]][roi_1,roi_2]
    if sum(~np.isnan(corr_mat_combined))>0:
        corr_mat_average = np.nanmean(corr_mat_combined)
    else:
        corr_mat_average = np.nan
    return corr_mat_average

    
def neuronList_to_neuronParts(neuronList, labels, parts='axon', proximal_r=20):
    if parts == None:
        return neuronList
    
    myList = pymaid.CatmaidNeuronList([])
    if hasattr(neuronList,'__len__'):
        for neuron in neuronList:
            myParts = cut_neuron(neuron, labels, parts, proximal_r = proximal_r)
            myList.neurons.append(myParts)
    else:
        neuron = neuronList 
        myParts = cut_neuron(neuron, labels, parts, proximal_r=proximal_r)
        myList.neurons.append(myParts)
    return myList

def cut_neuron_old(neuron, labels, parts='axon', soma_r = 30):
        # Define axon based on "axon" tag and count synapses
        myLabels = labels[ labels.skeleton_id == int(neuron.skeleton_id) ]
        myAxnLbl = myLabels[ myLabels.tag == 'axon']
        if len(myAxnLbl) < 1:
            print('WARNING: no axon tag on skel %s' % neuron.skeleton_id)
            myAxnStartID = None
        elif len(myAxnLbl) == 1:
            myAxnStartID = myAxnLbl.node_id.values
        else:
            print('WARNING: more than one axon tag on skel %s' % neuron.skeleton_id)
            myAxnStartID = myAxnLbl.node_id.values[0]
        if myAxnStartID is not None:
            if parts == 'axon':
                axon = neuron.prune_proximal_to(int(myAxnStartID), inplace=False)   
                return axon
            else: 
                myNeuron = neuron.prune_distal_to(int(myAxnStartID), inplace=False) 
                if myNeuron.soma == None:
                    print('neuron %s has no soma tag' %  myNeuron.skeleton_id)
                mySomaLoc = pymaid.get_node_location(myNeuron.soma)
                myDendrites = myNeuron
                all_nodes = myNeuron.nodes
                myProximalDendrite = navis.prune_at_depth(myDendrites,soma_r*1000)
                proximal_nodes = myProximalDendrite.nodes
                distal_nodes = all_nodes[~all_nodes.node_id.isin(proximal_nodes.node_id)]
                myDistalDendrite = navis.subset_neuron(myNeuron, distal_nodes)

                mySoma = navis.prune_at_depth(myDendrites,'5 microns')
                # 220325 update with apical tags
                         
                '''
                # Obselete:   
                apical_nodes = distal_nodes[distal_nodes.y < mySomaLoc.y[0] - soma_r*1000]
                basal_nodes = distal_nodes[distal_nodes.y > mySomaLoc.y[0]]
                myApicalDendrite = navis.subset_neuron(myNeuron,apical_nodes)
                myBasalDendrite = navis.subset_neuron(myNeuron, basal_nodes)
                '''
                # Apical and Basal
                myApLbl = myLabels[ myLabels.tag == 'apical dendrite']
                if len(myApLbl) < 1:
                    print('Warning, neuron %f has no apical dendrite label' % get_cid_from_skelID(neuron.skeleton_id)) 
                    myApicalDendrite = myDendrites # should probably return None or something else
                    myBasalDendrite = myDendrites
                elif len(myApLbl) == 1:
                    myApStartID = myApLbl.node_id.values
                    myBasalDendrite = myNeuron.prune_distal_to(myApStartID)
                    myBasalDendrite = navis.subset_neuron(myBasalDendrite, distal_nodes) # to avoid soma
                    basal_nodes = myBasalDendrite.nodes
                    apical_nodes = distal_nodes[~distal_nodes.node_id.isin(basal_nodes.node_id)]
                    #apical_nodes = [x for x in distal_nodes if not x in basal_nodes]
                    myApicalDendrite = navis.subset_neuron(myNeuron,apical_nodes)
                    myApicalDendrite = navis.subset_neuron(myApicalDendrite,distal_nodes) # to avoid soma
                else:
                    print('WARNING: more than one axon tag on skel %s' % neuron.skeleton_id)
                    myApStartID = myApLbl.node_id.values[0]
                    myBasalDendrite = myNeuron.prune_distal_to(myApStartID)
                    myBasalDendrite = navis.subset_neuron(myBasalDendrite, distal_nodes) # to avoid soma
                    basal_nodes = myBasalDendrite.nodes
                    apical_nodes = distal_nodes[~distal_nodes.node_id.isin(basal_nodes.node_id)]
                    #apical_nodes = [x for x in distal_nodes if not x in basal_nodes]
                    myApicalDendrite = navis.subset_neuron(myNeuron,apical_nodes)
                    myApicalDendrite = navis.subset_neuron(myApicalDendrite,distal_nodes) # to avoid soma
                if parts == 'dendrite':
                    return myDendrites
                elif parts == 'proximal':
                    return myProximalDendrite
                    #return mySoma
                elif parts == 'distal':
                    return myDistalDendrite
                elif parts == 'basal':
                    return myBasalDendrite
                elif parts == 'apical':
                    return myApicalDendrite
                else:
                    print('ERROR: parts input not recognized. Must be axon or dendrite')
        else: 
            myParts = neuron # need better handling of no axon - should still cut dendrites correctly
        return myParts

def get_tag_id(neuron, labels, tag):
        myLabels = labels[labels.skeleton_id == int(neuron.skeleton_id) ]
        myAxnLbl = myLabels[ myLabels.tag == tag]
        if len(myAxnLbl) < 1:
            print('WARNING: no %s tag on skel %s' % (tag, neuron.skeleton_id))
            myAxnStartID = None
        elif len(myAxnLbl) == 1:
            myAxnStartID = myAxnLbl.node_id.values
        else:
            print('WARNING: more than one axon tag on skel %s' % neuron.skeleton_id)
            myAxnStartID = myAxnLbl.node_id.values[0]
        return myAxnStartID

def cut_neuron(neuron, labels, parts='axon', proximal_r = 64):
    if parts == None:
        return neuron    

    # get axon tag and define axon/dendrite
    myAxnStartID = get_tag_id(neuron, labels, 'axon')
    if myAxnStartID is not None:
        axon = neuron.prune_proximal_to(int(myAxnStartID), inplace=False)   
        dendrites = neuron.prune_distal_to(int(myAxnStartID), inplace=False) 
    else:
        axon = None
        dendrites = neuron
    if parts == 'dendrite':
        return dendrites
    elif parts == 'axon':
        return axon

    # split dendrite into proximal (soma) / distal
    all_den_nodes = dendrites.nodes
    myProximalDendrite = navis.prune_at_depth(dendrites,proximal_r*1000)
    proximal_nodes = myProximalDendrite.nodes
    distal_nodes = all_den_nodes[~all_den_nodes.node_id.isin(proximal_nodes.node_id)]
    myDistalDendrite = navis.subset_neuron(dendrites, distal_nodes)

    if parts == 'proximal':
        return myProximalDendrite
    
    '''
    # distinguish proximal from soma
    mySoma = navis.prune_at_depth(myProximalDendrite, soma_r*1000)
    soma_nodes = mySoma.nodes
    #proximal_nodes = proximal_nodes[~proximal_nodes.node_id.isin(soma_nodes.node_id)]
    #myProximalDendrite = navis.subset_neuron(myProximalDendrite, proximal_nodes)
    if parts == 'soma':
        return myProximalDendrite
        #return mySoma
    if parts == 'proximal':
        return myProximalDendrite
    elif parts == 'distal':
        return myDistalDendrite
    '''

    # get apical dendrite tag and split apical/basal
    myApStartID = get_tag_id(neuron, labels, 'apical dendrite')
    if myApStartID is not None:
        myBasalDendrite = dendrites.prune_distal_to(myApStartID)
        myBasalDendrite = navis.subset_neuron(myBasalDendrite, distal_nodes)
        basal_nodes = myBasalDendrite.nodes
        apical_nodes = distal_nodes[~distal_nodes.node_id.isin(basal_nodes.node_id)]
        #apical_nodes = [x for x in distal_nodes if not x in basal_nodes]
        myApicalDendrite = navis.subset_neuron(myDistalDendrite,apical_nodes)
    else:
        myBasalDendrite = myDistalDendrite
        myApicalDendrite = None
    if parts == 'basal':
        return myBasalDendrite
    elif parts == 'apical':
        return myApicalDendrite
    else:
        print('ERROR: parts input not recognized, returning full neuron. Must be axon, dendrite, proxima, distal, basal, or apical')
        return neuron 

def adj_mat_to_edg_lst(x):
    ix_name = x.index.name if x.index.name else 'index'
    edges = x.reset_index(inplace=False,
                            drop=False).melt(id_vars=ix_name).values
    edg_lst = pd.DataFrame(edges, columns=['source','target','weight'])
    return edg_lst

def gen_potential_cns_df(mySources, myTargets, conv_cns=False):
    adj_mat = pymaid.adjacency_matrix(mySources, targets=myTargets)
    if conv_cns: # replace adj matrix to shared cn matrix
        #adj_mat = adj_mat > 0 
        adj_mat = adj_mat.dot(adj_mat.T)
        adj_mat.rename_axis('source_A', axis='rows',inplace=True)
        adj_mat.rename_axis('source_B', axis='columns',inplace=True)
    #graph = navis.network2nx(adj_mat)
    #edg_lst = nx.to_pandas_edgelist(graph)
    edg_lst = adj_mat_to_edg_lst(adj_mat)
    edg_lst['source_type'] = get_cell_type(edg_lst.source.values)
    s = np.stack(mySources['soma_loc'].values)
    edg_lst['target_type'] = get_cell_type(edg_lst.target.values)
    t = np.stack(myTargets['soma_loc'].values)
    if conv_cns:
        dists = dist.cdist(s, s, 'euclidean')/1000 #units of microns
    else:
        dists = dist.cdist(s, t, 'euclidean')/1000 #units of microns
    edg_lst['soma_dist'] = dists.flatten()
    return edg_lst

def cn_rate_from_pot_cns(edg_lst, syn_count = 'syn_count', cn_type=None, label=None, soma_dist_bin=None, count_mult=False):
    cn_rate = pd.Series(name=cn_type)
    cn_rate['cn_type'] = cn_type
    cn_rate['label'] = label
    if count_mult:
        cn_rate['n_cns'] = sum(edg_lst[syn_count])
    else:
        cn_rate['n_cns'] = len(edg_lst[edg_lst[syn_count]>0])
    cn_rate['n_pairs'] = len(edg_lst)
    if soma_dist_bin is not None:
        cn_rate['soma_dist_bin'] = soma_dist_bin
    if len(edg_lst)>0:
        cn_rate['frac_cn'] = cn_rate['n_cns'] / cn_rate['n_pairs']
    else:
        cn_rate['frac_cn'] = 0
    return cn_rate

def is_syn_checked(cn_id):
    tags = pymaid.get_node_tags(cn_id,'CONNECTOR')
    if len(tags) > 0:
        if any('checked' in string for string in tags[str(cn_id)]):
            return True
        else:
            return False
    else: 
        return False