# Plot all skeletons from neuron dataframe
# Currently restricted to pyramidal cells
import sys
sys.path.insert(0, '../tracing_analysis')
import pymaid
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import ppc_analysis_functions.figure_plotting as figs
import os
import logging
logging.disable(sys.maxsize)

dataset = 'PPC'
type='pyramidal'
cwd = os.getcwd()
mySources = os.sep.join([cwd,'../analysis_dataframes/MN_DF_new_PPC.pkl'])
video = False
figsDir = os.sep.join([cwd,'fig_panels/'])

if type == 'pyramidal':
    #figsize = (8.5,11)
    figsize = (17,22)
elif type == 'non pyramidal':
    #figsize = (8.5,4)
    figsize = (17,8)
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
    figs.plot_neuron_array(mySources, figsize=figsize, saveFig = figsDir+type+'_neurons_PPC_new.pdf')
