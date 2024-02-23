# Plot all skeletons from neuron dataframe
# Currently restricted to pyramidal cells

import pymaid
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import ppc_analysis_functions.figure_plotting as figs

import logging, sys
logging.disable(sys.maxsize)

dataset = 'PPC'
type='non pyramidal'
mySources = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/MN_DF_PPC.pkl'
video = False
#mySources = 'local_data/pyrDF_V1.pkl'
figsDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/figures_working/neuron_morphologies/'
if dataset == 'PPC':
    if type == 'pyramidal':
        figsize = (8.5,11)
    elif type == 'non pyramidal':
        figsize = (8.5,4)
    rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc', 
        api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff')
    if 'mySources' in locals():
        with open(mySources, 'rb') as f:  
            mySources = pickle.load(f)
        mySources = mySources[mySources.type == type]
    else:
        mySkelIDs = pymaid.get_skids_by_annotation("new matched neuron",allow_partial=True)
    if video:
        figs.plot_neuron_array(mySources, saveFig = 'figsDir/vid/', vid=True)
    else:
        figs.plot_neuron_array(mySources, figsize=figsize, saveFig = figsDir+type+'neurons_PPC_new.pdf')
    #plt.show()

if dataset == 'V1':
    rm = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc',
        api_token='9afd2769efa5374b8d48cb5c52af75218784e1ff', project_id=31)
    if 'mySources' in locals():
        with open(mySources, 'rb') as f:  
            mySources = pickle.load(f)
        figs.plot_neuron_array(mySources, saveFig = figsDir+type+'neurons_V1.pdf')
    else:
        mySkelIDs = pymaid.get_skids_by_annotation("matched neuron",allow_partial=True)
        figs.plot_neuron_array(mySkelIDs, figsize=figsize, saveFig = figsDir+type+'_neurons_V1.pdf', annotTag = 'matched neuron')
    #plt.show()
    