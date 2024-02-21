import pymaid
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys
from datetime import date
import pickle
import time
import logging, sys
from scipy.spatial import distance as dist
import networkx as nx
from sklearn.linear_model import LinearRegression
import ppc_analysis_functions.catmaid_API as cAPI
import analysis_dataframes as myDF
from scipy import stats as stats
import navis
from matplotlib.offsetbox import AnchoredText
from scipy.stats import kruskal
from scipy.stats import f_oneway
import random
from matplotlib import cm
import statsmodels
import statsmodels.api as sm
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import itertools
from seaborn.categorical import _ViolinPlotter

def plot_neuron(skel_id, ax = None, linewidth = 1, labels=None, view=('x', '-y'), plot_syn = False, scalebar=None, autoscale = False, s=20):
    if labels is None:
        labels = pymaid.get_label_list()
    plotDict = {}
    myNeuron = pymaid.get_neuron(int(skel_id))
    myLabels = labels[labels.skeleton_id == int(skel_id)]
    plotDict['myNeuron'] = myNeuron
    #(plotDict['myAxon'], plotDict['myAxonTrunk'],
    #    plotDict['myAxonCollaterals'], plotDict['myMyelination']
    #) = cAPI.get_axon_components(myNeuron, myLabels)
    plotDict['myAxon'] = cAPI.cut_neuron(myNeuron, labels, parts='axon')
    plotDict['myProximal'] = cAPI.cut_neuron(myNeuron, labels, parts='proximal')
    plotDict['myApical'] = cAPI.cut_neuron(myNeuron, labels, parts='apical')
    plotDict['myBasal'] = cAPI.cut_neuron(myNeuron, labels, parts='basal')

    myCnPre = pymaid.get_connectors(plotDict['myNeuron'],relation_type='presynaptic_to')
    if ax is not None:
        navis.plot2d(plotDict['myNeuron'],ax=ax,connectors=False, 
                linewidth=linewidth,method='2d',color='blue',scalebar=scalebar,view=view)
    else:
        fig,ax = navis.plot2d(plotDict['myNeuron'],connectors=False, 
                linewidth=linewidth,method='2d',color='blue',scalebar=scalebar,view=view)
    try:
        if plotDict['myAxon']:
            navis.plot2d(plotDict['myAxon'],ax=ax,method='2d',autoscale=autoscale,
                    connectors=False,linewidth=linewidth,color='blue',view=view)
        # ATK 231128 plot all dendrite as orange
        if plotDict['myProximal']:
            navis.plot2d(plotDict['myProximal'],ax=ax,method='2d',autoscale=autoscale,
                    connectors=False,linewidth=linewidth,color='orange',view=view) # HACK change to orange for inhib
        if plotDict['myApical']:
            navis.plot2d(plotDict['myApical'],ax=ax,method='2d',autoscale=autoscale,  
                    connectors=False,linewidth=linewidth,color='orange',view=view)
        if plotDict['myBasal']:
            navis.plot2d(plotDict['myBasal'],ax=ax,method='2d',autoscale=autoscale,  
                    connectors=False,linewidth=linewidth,color='orange',view=view)
        '''
        if plotDict['myAxonTrunk']:
            navis.plot2d(plotDict['myAxonTrunk'],ax=ax,method='2d',
                    connectors=False,linewidth=linewidth,color='red',view=view)

        if plotDict['myAxonCollaterals']:
            navis.plot2d(plotDict['myAxonCollaterals'],ax=ax,method='2d',
                linewidth=linewidth,color='cyan',view=view)
        if plotDict['myMyelination']:
            navis.plot2d(plotDict['myMyelination'],ax=ax,method='2d',
                linewidth=3,color='black',view=view)
        '''
        if len(myCnPre)>0 and plot_syn:
            navis.plot2d(myCnPre[['x','y','z']].values, ax=ax, method='2d',
            scatter_kws={'s': s,'color': 'cyzn','marker': 'D'},view=view)
    except: 
        print('error in figure_plotting.plot_neuron')
    return ax

def plot_neuron_array(mySources, labels=None, saveFig=False, vid=False, 
        annotTag = 'new matched neuron', figsize = (8.5,11)):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.animation as animation
    # Dump neuron objects into dict
    pymaid.clear_cache()
    if labels is None:
        labels = pymaid.get_label_list()
    plotDict = {}

    if isinstance(mySources, pd.DataFrame):
        skel_ids = mySources.skeleton_id.values
    else: 
        skel_ids = mySources
    if not vid:
        # Plot in array of subplots
        plt.subplots(figsize=figsize)
        plt.subplots_adjust(hspace = 0.5, wspace = 0)
        n_rows = int(np.ceil(len(skel_ids)**.5))
        n_columns = int(np.ceil(len(skel_ids)**.5))

        #n_columns = 8 # 231128 ATK hardcode for fig
        #n_columns = 12
        plt.subplots_adjust(wspace=0, hspace = 0.6)
        n_rows = int(np.ceil(len(skel_ids)/n_columns))
        for (i,skel_id) in enumerate(skel_ids):
            print('plotting skel_id %s' % skel_id  )
            ax = plt.subplot(n_rows,n_columns,i+1)
            if isinstance(mySources, pd.DataFrame):
                plt.title(int(skel_id), fontsize=16,pad=-50)
            plot_neuron(skel_id,ax=ax, scalebar='50 um')
            plt.axis('square')
            #plt.axis('on')
        if saveFig:
            plt.savefig(saveFig,dpi=600,bbox_inches='tight')
    else:
        img = [] # some array of images
        frames = [] # for storing the generated images
        fig = plt.figure()
        for (i,skel_id) in enumerate(skel_ids):
            print('plotting skel_id %s' % skel_id  )
            if isinstance(mySources, pd.DataFrame):
                plt.title('MN %0.1f' % mySources.iloc[i].matched_cell_ID)
            else:
                plt.title('MN %0.1f' % cAPI.get_cid_from_skelID(skel_id, annotTag=annotTag))
            plot_neuron(plotDict[skel_id])
            plt.axis('square')
            if saveFig:
                plt.savefig(saveFig+'file%d.png' % i)
            #plt.axis('on')
         
def plotTracingStatus(plotNeuronsDf, xlabel='MN_id'):
    unknown = plotNeuronsDf.num_synout
    glia = plotNeuronsDf.num_synout_uncertain+plotNeuronsDf.num_synout_outside+plotNeuronsDf.num_synout_soma + plotNeuronsDf.num_synout_glia
    uncertain = plotNeuronsDf.num_synout_uncertain+plotNeuronsDf.num_synout_outside+plotNeuronsDf.num_synout_soma
    outside = plotNeuronsDf.num_synout_outside+plotNeuronsDf.num_synout_soma
    soma = plotNeuronsDf.num_synout_soma

    if xlabel=='MN_id':
        chart = sns.barplot(x='matched_cell_ID',y='num_synout',data=plotNeuronsDf,color = 'red')
        chart = sns.barplot(x='matched_cell_ID',y=glia,data=plotNeuronsDf,color='purple')
        chart = sns.barplot(x='matched_cell_ID',y=uncertain,data=plotNeuronsDf,color='orange')
        chart = sns.barplot(x='matched_cell_ID',y=outside,data=plotNeuronsDf,color='blue')
        chart = sns.barplot(x='matched_cell_ID',y=soma,data=plotNeuronsDf,color='cyan')
    else:
        chart = sns.barplot(x='skeleton_id',y='num_synout',data=plotNeuronsDf,color = 'red')
        chart = sns.barplot(x='skeleton_id',y=glia,data=plotNeuronsDf,color='purple')
        chart = sns.barplot(x='skeleton_id',y=uncertain,data=plotNeuronsDf,color='orange')
        chart = sns.barplot(x='skeleton_id',y=outside,data=plotNeuronsDf,color='blue')
        chart = sns.barplot(x='skeleton_id',y=soma,data=plotNeuronsDf,color='cyan')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

    bar1 = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bar2 = plt.Rectangle((0,0),1,1,fc='purple',  edgecolor = 'none')
    bar3 = plt.Rectangle((0,0),1,1,fc='orange',  edgecolor = 'none')
    bar4 = plt.Rectangle((0,0),1,1,fc='blue',  edgecolor = 'none')
    bar5 = plt.Rectangle((0,0),1,1,fc='cyan',  edgecolor = 'none')
    l = plt.legend([bar1, bar2, bar3, bar4, bar5], ['unknown', 'glia','uncertain continuation',
        'soma outside volume','found soma'], loc=1, ncol = 1, prop={'size':12})
    l.draw_frame(True)

    chart.set(xlabel='Matched Neuron ID', ylabel='number of outgoing synapses')

def plot_pairs_vs_selectivities(dataDF, xmetric = 'RL_diff', ymetric='psd_area', 
    select_pairs='selectivity', target_type=None,add_fit=True, dataset='PPC', pts_scale=1):
    
    markers={'pyramidal':'^', 'apical dendrite': 'd', 'non pyramidal':'o','unknown':'.'}
    sns.set(rc={'figure.figsize':(8,4)})
    sns.set(font_scale=1.5)
    
    pair_order=['Same','Different']
    type_order=['pyramidal','non pyramidal']
    type_labels=['pyr','non-pyr']   
    colors = {'Same': 'green','Different': 'purple'}
    dot_color = 'k'
    def fit_line(dataDF, xmetric, ymetric):
        X = dataDF[xmetric].values
        y = dataDF[ymetric].values
        X_ = X.reshape(-1, 1)
        y_ = y.reshape(-1, 1)
        reg = LinearRegression().fit(X_, y_)
        y_pred = reg.predict(X_)
        return y_pred

    def plot_select_diff_vs_psd(dataDF, xmetric='RL_diff',ymetric=ymetric,hue=select_pairs, style=target_type):
        g = sns.scatterplot(data=dataDF, x = xmetric, y=ymetric,hue=hue,style=style,markers=markers,
        s=20*pts_scale,edgecolor=None,palette=colors)
        y_pred = fit_line(dataDF, xmetric, ymetric)
        if add_fit:
            plt.plot(dataDF[xmetric].values,y_pred)
        plt.legend([],[], frameon=False)
        print(stats.spearmanr(dataDF[xmetric],b=dataDF[ymetric],nan_policy='omit'))
    def plot_select_class_vs_psd(dataDF, xclass=select_pairs, ymetric='psd_area', hue=None):
        hue_order = None
        
        if hue == xclass:
            hue=None
            order=pair_order
            print("t-test for all types")
            same_df = dataDF[dataDF[select_pairs] == 'Same']
            not_same_df = dataDF[dataDF[select_pairs] == 'Different']
            print(stats.ttest_ind(same_df[ymetric].values, not_same_df[ymetric].values, equal_var=False, nan_policy='omit',alternative='greater'))
        elif xclass == target_type:
            order=type_order
            hue_order=pair_order
            print("t-test for target types separately")
            for i in type_order:
                same_df = dataDF[dataDF.target_type==i][dataDF[select_pairs] == 'Same']
                not_same_df = dataDF[dataDF.target_type==i][dataDF[select_pairs] == 'Different']
                print("t-test for %s targets" % i)
                print(stats.ttest_ind(same_df[ymetric].values, not_same_df[ymetric].values, equal_var=False, nan_policy='omit',alternative='greater'))
        else:
            order=None
        g = sns.swarmplot(data=dataDF, x=xclass, y=ymetric,color=dot_color,hue=hue,
            hue_order=hue_order,dodge=True,order=order,s=3*pts_scale) 
        sns.boxplot(data=dataDF, x = xclass, y=ymetric,hue=hue, order=order, hue_order=hue_order, palette=colors, whis=np.inf)
        if xclass == target_type:
            g.axes.set_xticklabels(type_labels)
        g.axes.get_yaxis().set_visible(False)
        plt.legend([],[], frameon=False)
    def subsume_apical(target_type):
        if target_type == 'pyramidal' or target_type == 'apical dendrite':
            return 'pyramidal'
        elif target_type == 'non pyramidal':
            return 'non pyramidal'
        else:
            return 'unknown'
    plt.subplot(1,3,1)
    plot_select_diff_vs_psd(dataDF, xmetric=xmetric,ymetric=ymetric,hue=select_pairs)
    if dataset=='V1':
        dataDF['target_type'] = dataDF.apply (lambda row:subsume_apical(row.target_type), axis=1)
    plt.subplot(1,3,2)
    plot_select_class_vs_psd(dataDF, xclass=select_pairs, ymetric=ymetric, hue=select_pairs)
    plt.subplot(1,3,3)
    plot_select_class_vs_psd(dataDF, xclass=target_type, ymetric=ymetric, hue=select_pairs)

def plot_avg_conv_psd_vs_select(conv_pair_df):
    # Plot avg conv psd area vs selectivity pair types

    plt.subplots(figsize=(4,4))
    sns.set(font_scale=2)
    colors = palette ={"Same": "green", "Different": "purple"}

    sns.boxplot(data=conv_pair_df, 
            x='pair_selectivity_ROC',y='avg_conv_psd_area', palette = colors, width = 0.5)
    sns.swarmplot(data=conv_pair_df,
                    x='pair_selectivity_ROC',y='avg_conv_psd_area',color='black',alpha=1,s=3)
    #plt.ylim((0,0.5))
    plt.ylabel('Avg PSD Size ($\mu m^2$)')
    plt.xlabel('Choice Selectivity')

    same_df = conv_pair_df[conv_pair_df.pair_selectivity_ROC == 'Same']
    not_same_df = conv_pair_df[conv_pair_df.pair_selectivity_ROC == 'Different']
    print(len(same_df))
    print(len(not_same_df))
    print(stats.ttest_ind(same_df.avg_conv_psd_area, not_same_df.avg_conv_psd_area, equal_var=False, nan_policy='omit'))

def fit_line(dataDF, xmetric, ymetric):
    X = dataDF[xmetric].values
    y = dataDF[ymetric].values
    X_ = X.reshape(-1, 1)
    y_ = y.reshape(-1, 1)
    reg = LinearRegression().fit(X_, y_)
    y_pred = reg.predict(X_)
    return y_pred

def plot_pairs_vs_select_types(dataDF, xmetric = 'RL_diff', ymetric='psd_area', 
    select_pairs='selectivity', target_type=None,add_fit=True, dataset='PPC', pts_scale=1, select_types='opp'):
    
    markers={'pyramidal':'^', 'apical dendrite': 'd', 'non pyramidal':'o','unknown':'.'}
    sns.set(rc={'figure.figsize':(8,10)})
    sns.set(font_scale=1.5)

    if select_types=='samediff':
        pair_order=['Same','Different']
        colors = {'Same': 'green','Different': 'purple'}  
    elif select_types=='opp':
        pair_order=['Same','Non-selective','Opposite']
        colors = {'Same': 'green','Different': 'purple', 'Opposite': 'purple','Non-selective':'gray'}
    type_order=['pyramidal','non pyramidal']
    type_labels=['pyr','non-pyr']   
    dot_color = 'k'
    def fit_line(dataDF, xmetric, ymetric):
        nas = np.logical_or(np.isnan(dataDF[xmetric]), np.isnan(dataDF[ymetric]))
        X = dataDF[~nas][xmetric].values
        y = dataDF[~nas][ymetric].values
        X_ = X.reshape(-1, 1)
        y_ = y.reshape(-1, 1)
        if len(X_)>1:
            reg = LinearRegression().fit(X_, y_)
            y_pred = reg.predict(X_)
        else: 
            y_pred = y_
        return y_pred

    def plot_select_diff_vs_psd(dataDF, xmetric='RL_diff',ymetric=ymetric,hue=select_pairs, style=target_type):
        g = sns.scatterplot(data=dataDF, x = xmetric, y=ymetric,hue=hue,style=style,markers=markers,
        s=20*pts_scale,edgecolor=None,palette=colors)
        y_pred = fit_line(dataDF, xmetric, ymetric)
        if add_fit:
            plt.plot(dataDF[xmetric].values,y_pred)
        plt.legend([],[], frameon=False)
        print('Spearmanr:')
        print(stats.spearmanr(dataDF[xmetric],b=dataDF[ymetric],nan_policy='omit'))
        print('Pearsonr:')
        print(stats.pearsonr(dataDF[xmetric],dataDF[ymetric]))
   
    def plot_select_class_vs_psd(dataDF, xclass=select_pairs, ymetric='psd_area', hue=None):
        hue_order = None
        hue=None
        order=pair_order
        same_df = dataDF[dataDF[select_pairs] == 'Same']
        same_df = same_df[~np.isnan(same_df[ymetric])]
        not_same_df = dataDF[dataDF[select_pairs] == 'Different']
        not_same_df = not_same_df[~np.isnan(same_df[ymetric])]

        print(stats.ttest_ind(same_df[ymetric].values, not_same_df[ymetric].values, 
            equal_var=False, nan_policy='omit',alternative='two-sided'))
        g = sns.swarmplot(data=dataDF, x=xclass, y=ymetric,color=dot_color,hue=hue,
            hue_order=hue_order,dodge=True,order=order,s=3*pts_scale) 
        sns.boxplot(data=dataDF, x = xclass, y=ymetric,hue=hue, order=order, hue_order=hue_order, palette=colors, whis=np.inf)
        if xclass == target_type:
            g.axes.set_xticklabels(type_labels)
        g.axes.get_yaxis().set_visible(False)
        plt.legend([],[], frameon=False)
    def subsume_apical(target_type):
        if target_type == 'pyramidal' or target_type == 'apical dendrite':
            return 'pyramidal'
        elif target_type == 'non pyramidal':
            return 'non pyramidal'
        else:
            return 'unknown'
    print('E-to-E')
    E2E_DF = dataDF[dataDF.target_type.isin(['pyramidal','apical dendrite'])]
    print('%i -to-E connections, %i same, %i diff' % 
        (len(E2E_DF),len(E2E_DF[E2E_DF[select_pairs]=='Same']),len(E2E_DF[E2E_DF[select_pairs]=='Different'])))
    plt.subplot(2,2,1)
    plot_select_diff_vs_psd(dataDF[dataDF.target_type.isin(['pyramidal', 'apical dendrite'])], xmetric=xmetric,ymetric=ymetric)
    if dataset=='V1':
        dataDF['target_type'] = dataDF.apply (lambda row:subsume_apical(row.target_type), axis=1)
    plt.subplot(2,2,2)
    plot_select_class_vs_psd(dataDF[dataDF.target_type=='pyramidal'], xclass=select_pairs, ymetric=ymetric, hue=select_pairs)

    print('-to-I')
    E2I_DF = dataDF[dataDF.target_type=='non pyramidal']
    print('%i -to-I connections, %i same, %i diff' % 
        (len(E2I_DF),len(E2I_DF[E2I_DF[select_pairs]=='Same']),len(E2I_DF[E2I_DF[select_pairs]=='Different'])))
    plt.subplot(2,2,3)
    plot_select_diff_vs_psd(dataDF[dataDF.target_type=='non pyramidal'], xmetric=xmetric,ymetric=ymetric)
    plt.subplot(2,2,4)
    plot_select_class_vs_psd(dataDF[dataDF.target_type=='non pyramidal'], xclass=select_pairs, ymetric=ymetric, hue=select_pairs)

def plot_epochs_metrics(data_DF, n_shuf=0, ymetric='avg_conv_psd_area', useMedian=False):
    block_shuf_DF = myDF.gen_block_conv_shuf(data_DF, ymetric=ymetric, n_shuf=n_shuf, useMedian=useMedian)
    block_labels = ('cueEarly','cueLate','delayEarly','delayTurn','turnITI')
    block_sel_labels = ['pair_selectivity_'+ i for i in block_labels]
    for block in block_labels:
        block_data = block_shuf_DF[block_shuf_DF.block==block]
        data_psd_frac = block_data[block_data.shuffle=='data'].psd_frac.values[0]
        shuf_psd_frac = block_data[block_data.shuffle=='shuffle'].psd_frac.values
        perc = stats.percentileofscore(shuf_psd_frac, data_psd_frac)
        print(block+' percentile is %f' % perc)
    plt.subplots(figsize=(6,4))
    sns.set(font_scale=1.5)  
    g=sns.pointplot(data=block_shuf_DF[block_shuf_DF.shuffle == 'data'], x='block',y='psd_frac',color='g',linestyles='-', alpha=.5)
    sns.lineplot(data=block_shuf_DF[block_shuf_DF.shuffle == 'shuffle'], x='block',y='psd_frac',
        ci='sd', color='k',linestyle='--')

    plt.legend([],[], frameon=False)
    g.get_xaxis().set_visible(False)

def plot_scatter(dataDF, xmetric='corr_raw',ymetric='psd_area',hue=None, 
style=None, markers=None, pts_scale=1, colors=None, add_fit=True):
    g = sns.scatterplot(data=dataDF, x = xmetric, y=ymetric,hue=hue,style=style,markers=markers,
    s=20*pts_scale,edgecolor=None,palette=colors)
    y_pred = fit_line(dataDF, xmetric, ymetric)
    if add_fit:
        plt.plot(dataDF[xmetric].values,y_pred)
    plt.legend([],[], frameon=False)
    c,p = stats.spearmanr(dataDF[xmetric],b=dataDF[ymetric],nan_policy='omit')
    plt.title('R = %f' % c)
    print(stats.spearmanr(dataDF[xmetric],b=dataDF[ymetric],nan_policy='omit'))

def scatter(dataDF,src_type = None, tgt_type = None, x='pair_select_idx', y='psd_area',hue=None, hue_order=None, errs = 1, s = 10,
    palette='PRGn',edgecolor='black',xlim=None, ylim=None,size=None, sig_test='pearson',y_log_scale=False, x_log_scale=False,
    style=None, markers=None, plot_errs = False, make_plot=True, ax=None, color='k', fit_color='red',add_fit=True):

    dataDF = src_tgt_type(dataDF, src_type, tgt_type)
    if make_plot:
        if ax is None:
            ax=sns.scatterplot(data=dataDF, x=x, y=y,hue=hue, hue_order=hue_order,palette=palette,edgecolor=edgecolor, 
            size=size, s=s, style=style, markers=markers, color=color)
        else:
            sns.scatterplot(data=dataDF, x=x, y=y,hue=hue, hue_order=hue_order,palette=palette,edgecolor=edgecolor, 
            size=size, s=s, style=style, markers=markers,color=color, ax=ax)
        plt.margins(.1,.1)
        ax.set_clip_on(False)
        if x_log_scale:
            ax.set(xscale='log')
        if y_log_scale:
            ax.set(yscale='log')
        if xlim is None:
            xlim = ax.get_xlim()
        if ylim is None:
            ylim = ax.get_ylim()
        ax.set(xlim=xlim)
        ax.set(ylim=ylim)
        plt.legend([],[], frameon=False)

    #g=sns.regplot(data=dataDF, x=x, y=y,ci=ci,scatter=False, color='k')
    #drop na values
    dataDF = dataDF.dropna(axis = 0, subset = [x,y])
    (c,p) = (np.nan, np.nan)
    if len(dataDF) > 0:
        Y = dataDF[y].values
        X = dataDF[x].values
        X = sm.add_constant(X)
        wls_model = sm.WLS(Y, X, weights = 1)
        results = wls_model.fit()
        if sig_test == 'pearson_weighted_xerr':
            wls_model = sm.WLS(Y, X, weights = 1./np.square(dataDF[errs].values))
            results = wls_model.fit()
            if plot_errs:
                plt.errorbar(x=dataDF[x].values,y=dataDF[y].values,xerr=dataDF[errs].values, fmt='none', color='k',elinewidth=0.5, capsize=1)
            (c, p) = (results.rsquared, results.pvalues[1])
            anc = AnchoredText('c = %.2g' % c, loc="upper left", frameon=False)
        elif sig_test == 'pearson_weighted_yerr':
            wls_model = sm.WLS(Y, X, weights = 1./(dataDF[errs].values**2))
            results = wls_model.fit()
            if plot_errs:
                plt.errorbar(x=dataDF[x].values,y=dataDF[y].values,yerr=dataDF[errs].values, fmt='none', color='k',elinewidth=0.5, capsize=1)
            (c, p) = (results.coef[1], results.pvalues[1])
            anc = AnchoredText('c = %.2g' % c, loc="upper left", frameon=False)
        elif sig_test == 'spearman':
            (c,p) = stats.spearmanr(dataDF[x],b=dataDF[y],nan_policy='omit')
            #print(stats.spearmanr(dataDF[x],b=dataDF[y],nan_policy='omit'))
            anc = AnchoredText('c = %.2g' % c, loc="upper left", frameon=False)
        elif sig_test == 'pearson':
            nas = np.logical_or(np.isnan(dataDF[x]), np.isnan(dataDF[y]))
            (c,p) = stats.pearsonr(dataDF[x][~nas],dataDF[y][~nas])
            #print(stats.pearsonr(dataDF[x][~nas],dataDF[y][~nas]))
            anc = AnchoredText('c = %.2g' % c, loc="upper left", frameon=False)
    if make_plot and fit_color is not None:
        if ax is None:
            ax=plt.plot(xlim, results.predict(sm.add_constant(xlim)), '--', color=fit_color)
        else:
            ax.plot(xlim, results.predict(sm.add_constant(xlim)), '--', color=fit_color)       
        #print(results.predict(sm.add_constant(xlim)))
            #anc = AnchoredText('c = %.1g, p=%.1g' % (c,p), loc="upper left", frameon=False)
        #ax.add_artist(anc)
        plt.legend([],[], frameon=False)
    return (c,p)


def catplot(dataDF, src_type = None, tgt_type = None, x='pair_selectivity', y='psd_area',s=10,whis=np.inf,
    palette = ['purple','gray','green'], order=['Opp','Non','Same'], sig_test='f_test',swarm=True):
    dataDF = src_tgt_type(dataDF, src_type, tgt_type)
    dataDF = dataDF.dropna(axis = 0, subset = [x,y])
    if swarm:
        g=sns.swarmplot(data=dataDF, x=x, y=y,color='k',order=order, s=s)
    g=sns.boxplot(data=dataDF, x=x, y=y,order=order, palette=palette, whis=whis, width = 0.7)
    plt.legend([],[], frameon=False)

    if sig_test == 'f_test':
        (s, p) = f_oneway(dataDF[dataDF.pair_selectivity == 'Same'][y], 
        dataDF[dataDF.pair_selectivity == 'Non'][y], 
        dataDF[dataDF.pair_selectivity == 'Opp'][y])   
        anc = AnchoredText('p=%.1g' % p, loc="upper left", frameon=False)
        g.add_artist(anc)
    elif sig_test == 'f_test_LNR':
        (s, p) = f_oneway(dataDF[dataDF[x] == 'Left'][y], 
        dataDF[dataDF[x] == 'Non'][y], 
        dataDF[dataDF[x] == 'Right'][y])   
        anc = AnchoredText('p=%.1g' % p, loc="upper left", frameon=False)
        g.add_artist(anc)
    elif sig_test == 'f_test_SameOpp':
        (s, p) = f_oneway(dataDF[dataDF.pair_selectivity == 'Same'][y], 
        dataDF[dataDF.pair_selectivity == 'Opp'][y])   
        anc = AnchoredText('p=%.1g' % p, loc="upper left", frameon=False)
        g.add_artist(anc)
    elif sig_test == 'K-W':
        try:
            if sum(dataDF.pair_selectivity == 'Non')==0:
                (s, p) = kruskal(dataDF[dataDF.pair_selectivity == 'Same'][y], 
                dataDF[dataDF.pair_selectivity == 'Opp'][y])  
            else:   
                (s, p) = kruskal(dataDF[dataDF.pair_selectivity == 'Same'][y], 
                dataDF[dataDF.pair_selectivity == 'Non'][y], 
                dataDF[dataDF.pair_selectivity == 'Opp'][y])     
            anc = AnchoredText('p=%.1g' % p, loc="upper left", frameon=False)
            g.add_artist(anc) 
        except:
            print('error in K-W test')
            (s,p) = (np.nan, np.nan)
    
def cn_rates(dataDF, cn_type=None, x='label', y='frac_cn', num='n_cns', denom = 'n_pairs', shuf='shuf',hue='shuf',
    palette = ['green','gray','purple'], order=['Same','Non','Opp'], cis = None):
    if cn_type is not None:
        dataDF = dataDF[dataDF.cn_type==cn_type]
    if shuf is not None:
        dataDF_shuf = dataDF
        dataDF = dataDF[dataDF[shuf]=='data']
        g = sns.barplot(data=dataDF_shuf, x=x,y=y,order=order,palette=palette, hue=hue)
        g = sns.barplot(data=dataDF_shuf, x=x,y=y,order=order,palette=palette, ci='sd', alpha=0, hue=hue)
        plt.bar_label(g.containers[0],labels=['%i/%i' % (a,b) for (a,b) in zip(dataDF['num'], dataDF['denom'])],size=16,label_type='edge')
    #sns.boxplot(data=dataDF[dataDF.shuf=='shuf'], x=x,y=y,width=0.5,whis=1,order=order, saturation=0.5)
    else:
        #g = sns.barplot(data=dataDF, x=x,y=y,order=order,palette=palette)
        if cis is not None:
            my_errs = [(dataDF[cis].values[i]-dataDF[y].values[i]) for i in range(len(dataDF))]
            my_errs = np.abs(np.array([np.array(i) for i in my_errs]).transpose())
        else:
            my_errs = None
        g = plt.bar(dataDF[x], dataDF[y], color=palette, yerr=my_errs)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.bar_label(g,labels=['%i/%i' % (a,b) for (a,b) in zip(dataDF['num'], dataDF['denom'])],size=16,label_type='edge')

def src_tgt_type(dataDF, src_type=None, tgt_type=None):
    if src_type is not None:
        if type(src_type)==str:
            dataDF = dataDF[dataDF.source_type == src_type]
        else:
            dataDF = dataDF[dataDF.source_type.isin(src_type)]
    if tgt_type is not None:
        if type(tgt_type)==str:
            dataDF = dataDF[dataDF.target_type == tgt_type]
        else:
            dataDF = dataDF[dataDF.target_type.isin(tgt_type)]
    return dataDF

def calc_syn_size_correlations(my_cnDF, source_type=None, target_type=None, n_rand = 200, log_scale=False,ylim = None, sig_test = 'pearson', make_plot=False):
    if target_type is not None:
        if isinstance(source_type,list):
            my_cnDF = my_cnDF[my_cnDF['source_type'].isin(source_type)]
            src_label = source_type[0]
        else:
            my_cnDF = my_cnDF[my_cnDF['source_type']==source_type] 
            src_label = source_type
    if target_type is not None:
        if isinstance(target_type,list):
            my_cnDF = my_cnDF[my_cnDF['target_type'].isin(target_type)]
            tgt_label = target_type[0]
        else:
            my_cnDF = my_cnDF[my_cnDF['target_type']==target_type]
            tgt_label = target_type

    # synapse pairs where both pre and post-synaptic neurons are the same
    dual_pairs_array = myDF.psd_pairs_from_grouped_syns(my_cnDF['psd_areas'].values)
    print('%i pre/post cns; %i pre/post syn pairs' % (len(my_cnDF), len(dual_pairs_array)))
    if log_scale:
        dual_pairs_array = np.log10(dual_pairs_array)
    shared_pre_post_syn_pairs = pd.DataFrame(dual_pairs_array, columns={'psd_area_A','psd_area_B'})

    if make_plot:
        plt.subplot(1,2,1)
        sns.set(rc={'figure.figsize':(6,4)}, font_scale = 1.5)
        plt.subplots_adjust(hspace = 0.6, wspace=0.4)
    (c_dual_pairs, p) = scatter(shared_pre_post_syn_pairs, x='psd_area_A', y='psd_area_B', sig_test = sig_test, make_plot=make_plot)
    print('Data corr = %.3f' % c_dual_pairs)
    # calc bootstrap of data
    c_boots = np.zeros(n_rand)
    for i in range(n_rand):
        boot_df = shared_pre_post_syn_pairs.sample(frac=1, replace=True, axis='rows')
        (c_boots[i], p) = scatter(boot_df, x='psd_area_A', y='psd_area_B', sig_test = sig_test, make_plot=False) 

    '''
    # synapse pairs where only pre-synaptic neurons are the same
    my_cnDF_src_grp = my_cnDF.groupby('source').aggregate({'psd_areas': lambda x: tuple(x)})
    #  pick only one syn from each pre-post pair
    src_grouped_psd_areas = [[random.choice(cn) for cn in src_row] for src_row in my_cnDF_src_grp['psd_areas']]
    pairs_array = myDF.psd_pairs_from_grouped_syns(src_grouped_psd_areas)
    print('%i pre neurons; %i pre syn pairs' % (len(my_cnDF_src_grp ), len(pairs_array)))
    if log_scale:
        pairs_array = np.log10(pairs_array)
    shared_pre_syn_pairs = pd.DataFrame(pairs_array, columns={'psd_area_A','psd_area_B'})
    # synapse pairs where only post-synpatic neurons are the same
    my_cnDF_tgt_grp = my_cnDF.groupby('target').aggregate({'psd_areas': lambda x: tuple(x)})
    tgt_grouped_psd_areas = [[random.choice(cn) for cn in tgt_row] for tgt_row in my_cnDF_tgt_grp['psd_areas']]
    pairs_array = myDF.psd_pairs_from_grouped_syns(tgt_grouped_psd_areas)
    print('%i post neurons; %i post syn pairs' % (len(my_cnDF_tgt_grp ), len(pairs_array)))
    if log_scale:
        pairs_array = np.log10(pairs_array)
    shared_post_syn_pairs = pd.DataFrame(pairs_array, columns={'psd_area_A','psd_area_B'})
    '''
    # randomly chosen synapse pairs
    c_rands = np.zeros(n_rand)
    for i in range(n_rand):
        n_pairs = len(dual_pairs_array)
        pairs_array = np.random.choice(np.concatenate(my_cnDF.psd_areas.values), size=(n_pairs,2))
        if log_scale:
            pairs_array = np.log10(pairs_array)
        random_syn_pairs = pd.DataFrame(pairs_array, columns={'psd_area_A','psd_area_B'})
        #print('%i rand syn pairs' % len(pairs_array))
        #sns.set(rc={'figure.figsize':(6,4)})
        #sns.set(font_scale=1.5)
        #plt.subplots_adjust(hspace = 0.4, wspace=0.4)
        #plt.subplot(1,2,1); plt.title('shared pre/post')
        (c_rands[i], p) = scatter(random_syn_pairs, x='psd_area_A', y='psd_area_B', sig_test = sig_test, make_plot=False)

    '''
    plt.subplot(1,5,2); plt.title('shared pre only')
    (c_pre, p) = scatter(shared_pre_syn_pairs, x='psd_area_A', y='psd_area_B', sig_test = sig_test)
    plt.subplot(1,5,3); plt.title('shared post only')
    (c_post, p) = scatter(shared_post_syn_pairs, x='psd_area_A', y='psd_area_B', sig_test = sig_test)
     plt.subplot(1,3,2); plt.title('random')
    (c_rand, p) = scatter(random_syn_pairs, x='psd_area_A', y='psd_area_B', sig_test = sig_test)

    c_compare = pd.DataFrame(); 
    #c_compare['corr'] = [c_pre_post, c_pre, c_post, c_rand]
    c_compare['corr'] = [c_pre_post,c_rand]
    #c_compare['labels'] = ['pre/post', 'pre','post','rand']
    c_compare['labels'] = ['pre/post','rand']
    plt.subplot(1,3,3); sns.barplot(data = c_compare, x='labels', y='corr')
    '''

    data_score = stats.percentileofscore(c_boots, 0, kind='rank')
    shuf_score = stats.percentileofscore(c_rands, c_dual_pairs, kind='rank')
    scores = dict({'data':data_score, 'shuf':shuf_score})
    #print('Data corr !=0: %.2f Data Diff from Shuf: %.2f' % (data_score, shuf_score))
    boot_corrs_DF = pd.DataFrame({'corrs':c_boots,'shuf':'data'})
    shuf_corrs_DF = pd.DataFrame({'corrs':c_rands,'shuf':'shuf'})
    my_corr_shuf_DF = pd.concat([boot_corrs_DF, shuf_corrs_DF])
    if make_plot:
        plt.subplot(1,2,2)
        sns.violinplot(data=my_corr_shuf_DF, x='shuf', y='corrs')
    if ylim is not None:
        plt.ylim(ylim)
    #plt.suptitle(src_label + '-to-' + tgt_label)
    return (shared_pre_post_syn_pairs, my_corr_shuf_DF, scores)

def plot_MN_Ca_activity(Ca_trial_means, tracingDf, cell_ID, title=None, mySessions = ['LD187_141216','LD187_141215','LD187_141214','LD187_141213','LD187_141212','LD187_141211',
    'LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206']):
    cmap = cm.get_cmap('viridis', len(mySessions))
    if isinstance(cell_ID, np.ndarray) or isinstance(cell_ID, list):
        num_cells = len(cell_ID)
        plt.rcParams["figure.figsize"] = [4,0.6*num_cells]
        fig, axs = plt.subplots(nrows=num_cells, ncols=2, sharex=True, sharey=True, squeeze=True)

        for cell_idx, cid in enumerate(cell_ID):
            sessions_ROI = tracingDf[tracingDf.matched_cell_ID == cid].sessions_ROI.values[0]
            myTitle=None
            if title is not None and len(title) == num_cells:
                myTitle = title[cell_idx]
            for idx, ROI in enumerate(sessions_ROI):
                if ROI > -1:
                    plot_session_Ca_activity(Ca_trial_means, mySessions[idx],ROI, title=myTitle, color=cmap(idx), zscore=True, axs=axs[cell_idx,:])
    else:
        plt.rcParams["figure.figsize"] = [4,0.6]
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True)
        sessions_ROI = tracingDf[tracingDf.matched_cell_ID == cell_ID].sessions_ROI.values[0]
        for idx, ROI in enumerate(sessions_ROI):
            if ROI > -1:
                plot_session_Ca_activity(Ca_trial_means, mySessions[idx],ROI, title=title, color=cmap(idx), zscore=True,axs=axs)

def plot_session_Ca_activity(Ca_trial_means, session, ROI, title=None, color = 'blocks', zscore=False, axs=None):
    trialTypes = ['wL_trials','bR_trials']
    wL_Ca_trialMean = Ca_trial_means[session]['wL_trials'][:, ROI]
    bR_Ca_trialMean = Ca_trial_means[session]['bR_trials'][:, ROI]

    if zscore:
        lr_concat = np.hstack((wL_Ca_trialMean,bR_Ca_trialMean))
        lr_concat_z = stats.zscore(lr_concat)
        wL_Ca_trialMean = np.split(lr_concat_z, 2)[0]
        bR_Ca_trialMean = np.split(lr_concat_z, 2)[1]
    if not axs.any():
        plt.rcParams["figure.figsize"] = [4.25,1]
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True)

    #axs[0].set_xlabel('left')   
    #axs[1].set_xlabel('right')
    axs[0].set_ylabel(title) 
    axs[0].set_yticklabels([])
    #ax.set_xticklabels([])

    for idx,trial_type in enumerate(trialTypes[:2]):
        if color == 'blocks':
            plot_Ca_activity_blocks(wL_Ca_trialMean,axs[0])
            plot_Ca_activity_blocks(bR_Ca_trialMean,axs[1])
        else:
            axs[0].plot(wL_Ca_trialMean, color=color, linewidth=0.3)
            axs[1].plot(bR_Ca_trialMean, color=color, linewidth=0.3)

def plot_Ca_activity_blocks(Ca_data_76,axs):
    # Define blocks
    # Currently set up to accept 63 time points (with ITI before removed)
    cueBlockEarly = np.arange(13,27,1)-13 #14:27; % 14 is running onset + 12 frames after
    cueBlockLate = np.arange(26,39,1)-13 #27:39; % 12 frames before cue offset (frame 39)
    delayBlockEarly = np.arange(38,52,1)-13 #39:52; % 39 is cue offset + 12 frames after
    delayTurnBlock = np.arange(51,65,1)-13 #52:65; % 12 frames before end of trial (turn a certain amt)
    turnITIblock = np.arange(64,76,1)-13 #65:76; % Trial end (reward given) an 12 frames after (dark, ITI)

    #Define colors
    cueEarlyColor = np.array([0, 161, 75])/255
    cueLateColor = np.array([255, 222, 23])/255
    delayEarlyColor = np.array([237, 28, 36])/255
    delayTurnColor = np.array([127, 63, 152])/255
    turnITIcolor = np.array([33, 64, 154])/255
    
    axs.plot(cueBlockEarly,Ca_data_76[cueBlockEarly],color=cueEarlyColor)
    axs.plot(cueBlockLate,Ca_data_76[cueBlockLate],color=cueLateColor)
    axs.plot(delayBlockEarly,Ca_data_76[delayBlockEarly],color=delayEarlyColor)
    axs.plot(delayTurnBlock,Ca_data_76[delayTurnBlock],color=delayTurnColor)
    axs.plot(turnITIblock,Ca_data_76[turnITIblock],color=turnITIcolor)

    return axs

# Calc shuffle of post-synaptic neuron selectivities
def compare_corr_vs_1Dshuf(data_df, x='select_idx_MI_abs', y='collat_syn_density', n_shuf=100, sig_test = 'spearman'):
    def calc_shuf_corr(df,sig_test=sig_test):
        df[x+'_shuf'] = df.sample(frac=1)[x].values
        (c,p) = scatter(df, x=x+'_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    def calc_bootstrap_data_corr(df,sig_test=sig_test):
        boot_df = df.sample(frac=1, replace=True, axis='rows')
        (c,p) = scatter(boot_df, x=x, y=y, sig_test=sig_test,make_plot=False)
        return c
    (data_corr,p) = scatter(data_df, x=x, y=y, sig_test=sig_test,make_plot=False)
    print(sig_test + ' Data Corr: %.3f' % data_corr)
    orig_data_df = pd.DataFrame({'corrs':data_corr, 'shuf':'data_orig'}, index=[0])

    boot_corrs_data = [calc_bootstrap_data_corr(data_df,sig_test = sig_test) for i in range(n_shuf)]
    boot_corrs_data_df = pd.DataFrame({'corrs':boot_corrs_data ,'shuf':'data'})
    boot_score = stats.percentileofscore(boot_corrs_data, 0, kind='rank')

    shuf_corrs= [calc_shuf_corr(data_df,sig_test = sig_test) for i in range(n_shuf)]
    shuf_corrs_df = pd.DataFrame({'corrs':shuf_corrs,'shuf':'shuf'})
    shuf_score = stats.percentileofscore(shuf_corrs, data_corr, kind='rank')

    #my_corr_shuf_DF = pd.concat([orig_data_df, boot_corrs_data_df, shuf_corrs_df]) 
    my_corr_shuf_DF = pd.concat([boot_corrs_data_df, shuf_corrs_df])
    scores = dict({'data':boot_score, 'shuf':shuf_score, 'c':data_corr})

    print('n = %i, bootstrap p = %.3f, shuf p = %.3f' % (len(data_df), boot_score, shuf_score))
    return (my_corr_shuf_DF, scores)

def calc_shuffle_comparisons_(data_df, x='pair_select_idx',y='syn_den', n_shuf=100, sig_test='pearson'):
    df = data_df
    df = df.dropna(axis = 0, subset = [x,y])
    (c,p) = scatter(df, x=x, y=y, sig_test=sig_test,make_plot=False)
    data_corr = c
    print('Data corr = %.3f' % c)
    print('t-test p = %.3f' % p)

    def calc_shuf_corr_both(df,sig_test=sig_test):
        df['source_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx.values, index = df.index)
        df['target_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx(row['source_select_idx_shuf'], row['target_select_idx_shuf']), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c

    shuf_corrs_both = [calc_shuf_corr_both(df,sig_test = sig_test) for i in range(n_shuf)]
    shuf_corrs_both_df = pd.DataFrame({'corrs':shuf_corrs_both,'shuf':'shuf'})
    both_score = stats.percentileofscore(shuf_corrs_both, data_corr, kind='rank')
    both_score = np.minimum(both_score, 100-both_score)/100
    data_df = pd.DataFrame({'corrs':data_corr, 'shuf':'data'}, index=[0])

    my_corr_shuf_DF = pd.concat([shuf_corrs_both_df, data_df])
    scores = dict({'c':data_corr,'shuf':both_score})
    print('n = %i, shuf p = %.3f' % (len(df), both_score))
    #print([data_score,source_score, target_score, both_score])
    return (my_corr_shuf_DF, scores)
def calc_shuffle_comparisons_new(data_df, x='pair_select_idx_new',y='syn_den', n_shuf=100, sig_test='pearson'):
    df = data_df
    df = df.dropna(axis = 0, subset = [x,y])
    (c,p) = scatter(df, x=x, y=y, sig_test=sig_test,make_plot=False)
    data_corr = c
    print('Data corr = %.3f' % c)
    print('t-test p = %.3f' % p)

    def calc_shuf_corr_both(df,sig_test='spearman'):
        df['source_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx.values, index = df.index)
        df['target_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx(row['source_select_idx_shuf'], row['target_select_idx_shuf']), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c

    def calc_shuf_corr_tmax(df,sig_test='spearman'):
        df['source_select_idx_t_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx_t.values, index = df.index)
        df['target_select_idx_t_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx_t.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx_t(row['source_select_idx_t_shuf'], row['target_select_idx_t_shuf'], type='max'), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    def calc_shuf_corr_tmax_filt(df,sig_test='spearman'):
        df['source_select_idx_t_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx_t_filt.values, index = df.index)
        df['target_select_idx_t_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx_t_filt.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx_t(row['source_select_idx_t_shuf'], row['target_select_idx_t_shuf'], type='max'), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    def calc_shuf_corr_tavg_filt(df,sig_test='spearman'):
        df['source_select_idx_t_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx_t_filt.values, index = df.index)
        df['target_select_idx_t_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx_t_filt.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx_t(row['source_select_idx_t_shuf'], row['target_select_idx_t_shuf'], type='avg'), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    def calc_shuf_corr(df, x='err_diff_pref', sig_test = 'spearman'):
        df[x+'_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows')[x].values, index = df.index)
        (c,p) = scatter(df, x=x+'_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    if x == 'pair_select_idx':
        shuf_corrs_both = [calc_shuf_corr_both(df,sig_test = sig_test) for i in range(n_shuf)]
    elif x == 'pair_select_idx_tmax':
        shuf_corrs_both = [calc_shuf_corr_tmax(df,sig_test = sig_test) for i in range(n_shuf)]
    elif x == 'pair_select_idx_tmax_filt':
        shuf_corrs_both = [calc_shuf_corr_tmax_filt(df,sig_test = sig_test) for i in range(n_shuf)]
    elif x == 'pair_select_idx_tavg_filt':
        shuf_corrs_both = [calc_shuf_corr_tavg_filt(df,sig_test = sig_test) for i in range(n_shuf)]
    else:
        shuf_corrs_both = [calc_shuf_corr(df,x=x,sig_test = sig_test) for i in range(n_shuf)]

    shuf_corrs_both_df = pd.DataFrame({'corrs':shuf_corrs_both,'shuf':'shuf'})
    both_score = stats.percentileofscore(shuf_corrs_both, data_corr, kind='rank')
    both_score = np.minimum(both_score, 100-both_score)/100
    data_df = pd.DataFrame({'corrs':data_corr, 'shuf':'data'}, index=[0])

    my_corr_shuf_DF = pd.concat([shuf_corrs_both_df, data_df])
    scores = dict({'c':data_corr,'shuf':both_score})
    print('n = %i, shuf p = %.3f' % (len(df), both_score))
    #print([data_score,source_score, target_score, both_score])
    return (my_corr_shuf_DF, scores)
'''
def calc_shuffle_comparisons(data_df, x='pair_select_idx',y='syn_den', n_shuf=100, sig_test='spearman'):
    df = data_df
    df = df.dropna(axis = 0, subset = [x,y])
    (c,p) = scatter(df, x=x, y=y, sig_test=sig_test,make_plot=False)
    data_corr = c
    print('Data corr = %.3f' % c)
    print('Quick est p = %.3f' % p)
    def calc_shuf_corr_target(df,sig_test=sig_test):
        df['target_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx(row['source_select_idx'], row['target_select_idx_shuf']), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    def calc_shuf_corr_source(df,sig_test=sig_test):
        df['source_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx(row['source_select_idx_shuf'], row['target_select_idx']), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    def calc_shuf_corr_both(df,sig_test=sig_test):
        df['source_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').source_select_idx.values, index = df.index)
        df['target_select_idx_shuf'] = pd.DataFrame(df.sample(frac=1, replace=False, axis='rows').target_select_idx.values, index = df.index)
        df['pair_select_idx_shuf'] =  df.apply (lambda row: myDF.add_pair_select_idx(row['source_select_idx_shuf'], row['target_select_idx_shuf']), axis=1)
        (c,p) = scatter(df, x='pair_select_idx_shuf', y=y, sig_test=sig_test,make_plot=False)
        return c
    def calc_bootstrap_data_corr(df,sig_test=sig_test):
        boot_df = df.sample(frac=1, replace=True, axis='rows')
        (c,p) = scatter(boot_df, x=x, y=y, sig_test=sig_test,make_plot=False)
        return c

    boot_corrs_data = [calc_bootstrap_data_corr(df,sig_test = sig_test) for i in range(n_shuf)]
    boot_corrs_data_df = pd.DataFrame({'corrs':boot_corrs_data ,'shuf':'data'})
    data_score = stats.percentileofscore(boot_corrs_data, 0, kind='rank')
    data_score = np.minimum(data_score, 100-data_score)/100

    shuf_corrs_source = [calc_shuf_corr_source(df,sig_test = sig_test) for i in range(n_shuf)]
    shuf_corrs_source_df = pd.DataFrame({'corrs':shuf_corrs_source,'shuf':'pre'})
    source_score = stats.percentileofscore(shuf_corrs_source, data_corr, kind='rank')
    source_score = np.minimum(source_score, 100-source_score)/100

    shuf_corrs_target = [calc_shuf_corr_target(df,sig_test = sig_test) for i in range(n_shuf)]
    shuf_corrs_target_df = pd.DataFrame({'corrs':shuf_corrs_target,'shuf':'post'})
    target_score = stats.percentileofscore(shuf_corrs_target, data_corr, kind='rank')
    target_score = np.minimum(target_score, 100-target_score)/100

    shuf_corrs_both = [calc_shuf_corr_both(df,sig_test = sig_test) for i in range(n_shuf)]
    shuf_corrs_both_df = pd.DataFrame({'corrs':shuf_corrs_both,'shuf':'shuf'})
    both_score = stats.percentileofscore(shuf_corrs_both, data_corr, kind='rank')
    both_score = np.minimum(both_score, 100-both_score)/100
    data_df = pd.DataFrame({'corrs':data_corr, 'shuf':'data'}, index=[0])

    #my_corr_shuf_DF = pd.concat([boot_corrs_data_df, shuf_corrs_source_df, shuf_corrs_target_df ,shuf_corrs_both_df, data_df])
    my_corr_shuf_DF = pd.concat([boot_corrs_data_df,shuf_corrs_both_df, data_df])
    scores = dict({'c':data_corr, 'data':data_score,'source':source_score, 'target':target_score, 'shuf':both_score})
    print('n = %i, bootstrap p = %.3f, shuf p = %.3f' % (len(df), data_score, both_score))
    #print([data_score,source_score, target_score, both_score])
    return (my_corr_shuf_DF, scores)
'''
def RL_colormap(select_idxs):
    my_LR_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:0000FF-43:4D4DFF-50:B3B3B3-57:FF4D4D-100:FF0000
    (0.000, (0.000, 0.000, 1.000)),
    (0.430, (0.302, 0.302, 1.000)),
    (0.500, (0.702, 0.702, 0.702)),
    (0.570, (1.000, 0.302, 0.302)),
    (1.000, (1.000, 0.000, 0.000))))
    #sat_val = .5
    #cmap = sns.diverging_palette(240, 10, l=60, s=99, center="dark", as_cmap=True)
    cmap = my_LR_cmap
    return [matplotlib.colors.rgb2hex(cmap((s_idx+1)/2)) for s_idx in select_idxs]


def pair_colormap(select_idxs):
    cmap =  LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#BA48CF-CCCCCC-2CB899
    (0.000, (0.729, 0.282, 0.812)),
    (0.500, (0.800, 0.800, 0.800)),
    (1.000, (0.173, 0.722, 0.600))))
    return [matplotlib.colors.rgb2hex(cmap(1/.2*(((s_idx+1)/2)-.5)+.5)) for s_idx in select_idxs]

def plot_syn_EM(center, dx = 2000, z_nudge=0, lower=0, upper=255):
    from skimage import exposure
    from pymaid.tiles import TileLoader
    #center = np.array(pymaid.get_connectors_between(mySkids[0], mySkids[1]).connector_loc[0])
    #znudge = 1
    center[2] = center[2]+40*z_nudge
    #dx = 2000 # increments of 40 nm only
    xrange = dx/1000
    min_co = center - dx/2
    max_co = center + dx/2
    #zoom_level = int(np.round(np.log2(dx/size/4)))
    zoom_level = 0
    print('zoom level = %i' % zoom_level)
    bbox = [c for co in zip(min_co[:2], max_co[:2]) for c in co] + [center[2]]
    print(bbox)

    job = TileLoader(bbox, stack_id = 43, zoom_level = zoom_level, coords='NM')
    job.load_in_memory()
    img = job.img[:,:,0]
    img = exposure.rescale_intensity(img, in_range=(lower,upper), out_range='uint8')
    print('I range is %i %i' % (np.min(img[:]), np.max(img[:])))
    fig = plt.imshow(img,cmap="gray", extent = [0, xrange, 0, xrange])
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    return img

def calc_corr_diffs(cn_DF, col, col_vals, pairs = None, x='pair_select_idx',y = 'syn_den'):
    my_corrs = {}
    for val in col_vals:
        (c,p) = scatter(cn_DF[cn_DF[col]==val], x=x, y=y, make_plot=False)
        my_corrs[val] = c
    my_corr_diffs = pd.DataFrame()
    
    if pairs is None:
        pairs = itertools.combinations(col_vals,2)
    for xy in pairs: #itertools.combinations(col_vals,2):
        corr_diff = my_corrs[xy[0]]-my_corrs[xy[1]]
        corr_df = pd.DataFrame(data={'pair':str(xy),'c': corr_diff}, index=[0])
        my_corr_diffs = pd.concat([my_corr_diffs, corr_df])
    return my_corr_diffs

def calc_corr_trend_reg(cn_DF, col, col_vals, pairs = None, x = 'pair_select_idx_new', y = 'syn_den', sig_test = 'pearson', return_fit_vals=False):
    my_corrs = {}
    for val in col_vals:
        (c,p) = scatter(cn_DF[cn_DF[col]==val], x=x, y=y, sig_test=sig_test, make_plot=False)
        my_corrs[val] = c
    inds = range(len(col_vals))
    corrs = [my_corrs[val] for val in col_vals]
    corrs_df = pd.DataFrame({'corrs':corrs, 'ind':inds})
    #print(corrs_df)

    slope, intercept, r_value, p_value, std_err = stats.linregress(inds,corrs)
    if return_fit_vals:
        return intercept + slope*inds
    #(reg, p) = scatter(corrs_df, x='ind', y='corrs', sig_test = 'pearson')
    #print(slope)
    else:
        return slope

def plot_corr_sigs(shuf_df_all, col, col_vals, n_shuf = 100, val_abv = None, ax=None):
    g=my_violinplot(data = shuf_df_all[shuf_df_all['shuf'].isin(['shuf'])], x=col, y='corrs', 
        ci='sd', order=col_vals, inner='quartiles', color = 'white', cut=0, scale='width', ax=ax)
    g.axhline(y=0, color='black', linestyle='-')
    for idx, val in enumerate(col_vals):
        my_shuf_df = shuf_df_all[shuf_df_all[col]==val]
        data_corr = my_shuf_df[my_shuf_df['shuf']=='data'].corrs.values[0]
        plt.plot([-.4+idx,.4+idx],[data_corr, data_corr], color='red', linewidth = 3)
    
    if val_abv is not None:
        g.set_xticklabels(val_abv)
    g.set_ylabel('Corr. Coef.')
    sns.despine()
    return g

def plot_corr_diffs(cn_DF_all, col, col_vals, n_shuf = 100, val_abv = None, x='pair_select_idx',y = 'syn_den', make_plot=True):

    data_diffs = calc_corr_diffs(cn_DF_all, col, col_vals, x = x, y = y)
    # Decide order of comparisons so all diffeerences are positive
    flips = data_diffs.c < 0

    pairs = [i for i in itertools.combinations(col_vals,2)]
    if val_abv is not None:
        pairs_abv = [i for i in itertools.combinations(val_abv,2)]
    for idx, flip in enumerate(flips):
        if flip:
            pairs[idx] = tuple(reversed(pairs[idx]))
            if val_abv is not None:
                pairs_abv[idx] = tuple(reversed(pairs_abv[idx]))
    # Recalculate with new orders
    data_diffs = calc_corr_diffs(cn_DF_all, col, col_vals, pairs=pairs, x=x, y = y)
    shuf_diffs_df = pd.DataFrame()
    for i in range(n_shuf):
        shuf_cn_DF = cn_DF_all
        shuf_cn_DF[col] = cn_DF_all.sample(frac=1)[col].values
        shuf = calc_corr_diffs(shuf_cn_DF, col,col_vals, pairs = pairs, x=x, y=y)
        shuf_diffs_df = pd.concat([shuf_diffs_df, shuf], axis=0, ignore_index=True)

    if make_plot:
        #sns.set(rc={'figure.figsize':(2,3)}, font_scale=1.5, style='ticks')
        g=my_violinplot(data=shuf_diffs_df, x='pair', y='c', color='white')
        sns.despine()
        g.set_xticklabels(g.get_xticklabels())#,rotation = 45)
        g.axhline(y=0, color='black', linestyle='-')
        if val_abv is not None:
            pairs_lbl = [str(pair[0])+'-'+str(pair[1]) for pair in pairs_abv]
            g.set_xticklabels(pairs_lbl)
    for idx,xy in enumerate(pairs): 
        data_diff = data_diffs[data_diffs.pair == str(xy)].c.values
        p = stats.percentileofscore(shuf_diffs_df[shuf_diffs_df.pair == str(xy)].c.values, data_diff)
        print('diff for %s = %f' % (str(xy), data_diff))
        print('p for %s = %f' % (str(xy), (100-p)/100))   
        if make_plot: 
            g.plot([-.4+idx,.4+idx],[data_diff, data_diff], color='red', linewidth = 3)
            g.set_ylabel('Rank Corr. Diff. ')
    if make_plot:
        return g
    else:
        return None

def plot_corr_trend_reg(cn_DF_all, col, col_vals, n_shuf = 100, sig_test='pearson',val_abv = None, x='pair_select_idx_new', y = 'syn_den'):
    data_reg = calc_corr_trend_reg(cn_DF_all, col, col_vals, sig_test=sig_test,x=x,y=y)
    shuf_regs = np.zeros(n_shuf)
    for i in range(n_shuf):
        shuf_cn_DF = cn_DF_all
        shuf_cn_DF[col] = cn_DF_all.sample(frac=1, replace=False)[col].values
        shuf_reg = calc_corr_trend_reg(shuf_cn_DF, col,col_vals, sig_test=sig_test, x=x,y=y)
        shuf_regs[i] = shuf_reg

    #sns.set(rc={'figure.figsize':(2,3)}, font_scale=1.5, style='ticks')
    g=my_violinplot(shuf_regs, color='white')
    sns.stripplot(shuf_regs, s=2)
    plt.axvline(x=data_reg)
    print(data_reg)
    perc = stats.percentileofscore(shuf_regs, data_reg)
    p = min(perc/100, (100-perc)/100)
    print('p = %f' % p)
    sns.despine()
    return shuf_regs

class MyVPlot(_ViolinPlotter):
    def draw_quartiles(self, ax, data, support, density, center, split=False, color='k'):
        """Draw the quartiles as lines at width of density."""
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        mean = np.mean(data)
        # ATK: draw mean instead
        self.draw_to_density(ax, center, mean, support, density, split,
                             linewidth=self.linewidth, color=color)

def my_violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
                  bw="scott", cut=0, scale="area", scale_hue=True, gridsize=100,
                  width=.8, inner="quartiles", split=False, dodge=True, orient=None,
                  linewidth=None, color=None, palette=None, saturation=.75,alpha=1,
                  ax=None, **kwargs):

    plotter = MyVPlot(x, y, hue, data, order, hue_order,
                      bw, cut, scale, scale_hue, gridsize,
                      width, inner, split, dodge, orient, linewidth,
                      color, palette, saturation)
    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    # change outlines to black 
    for i in ax.collections:
        i.set_edgecolor('k')
    for l in ax.lines:
        l.set_linestyle('-')
        l.set_color('black')
    return ax

def plot_cn_pair_3d_new(n1, n1_sidx, n2, n2_sidx, labels, plt_syn=True, cutoff_dist = None, syn_s = 20, fig_size=(6,4)):
    def get_skel_overlaps(n1, n2, min_overlap=5000):
        ix, dist = n2.snap(n1.nodes[['x','y','z']].values)
        n1.nodes['close'] = dist <= min_overlap
        return  navis.subset_neuron(n1, n1.nodes[n1.nodes['close']])

    if cutoff_dist is not None:
        navis.prune_at_depth(n1,cutoff_dist, inplace = True)
        navis.prune_at_depth(n2,cutoff_dist, inplace = True)
    (n1_axon,  myAxonTrunk, myAxonCollaterals, myMyelination) = cAPI.get_axon_components(n1, labels[labels.skeleton_id == int(n1.skeleton_id)])

    navis.resample_skeleton(n1_axon, '1 um', inplace=True)
    navis.smooth_skeleton(n1_axon, 2, inplace=True)

    n2_dendrite = cAPI.cut_neuron(n2, labels[labels.skeleton_id == int(n2.skeleton_id)], parts='dendrite')
    n1_overlap = get_skel_overlaps(n1_axon, n2_dendrite, min_overlap=5000)
    n1_overlap_context = get_skel_overlaps(n1_axon, n2_dendrite, min_overlap=100000)

    n1_soma_loc = pymaid.get_node_location(n1.soma[0])[['x','y','z']].values
    n1_soma = navis.prune_at_depth(n1, '7 um', inplace=False)

    n1_axon_soma = navis.stitch_skeletons([n1_axon, n1_soma])
    navis.resample_skeleton(n1_axon_soma, '1 um', inplace=True)
    navis.smooth_skeleton(n1_axon_soma, window=3, to_smooth=['x', 'y', 'z'], inplace=True)
    n1_axon_soma2 = n1_axon_soma.copy()

    n2_dendrite = cAPI.cut_neuron(n2, labels[labels.skeleton_id == int(n2.skeleton_id)], parts='dendrite')
    navis.resample_skeleton(n2_dendrite, '1 um', inplace=True)
    navis.smooth_skeleton(n2_dendrite, 2, inplace=True)
    n2_dendrite_thick = n2_dendrite.copy()
    navis.resample_skeleton(n2_dendrite_thick, '1 um', inplace=True)
    navis.smooth_skeleton(n2_dendrite_thick, 3, inplace=True)

    n1_overlap = get_skel_overlaps(n1_axon, n2_dendrite, min_overlap=5000)
    
    myCns = pymaid.get_connectors_between(n1_axon, n2_dendrite)
    cns = np.stack(myCns.connector_loc.values)

    n1_axon.nodes.radius = 250
    n2_dendrite_thick.nodes.radius = 500
    #n2_dendrite_thick.nodes.radius = 1000
    n1_overlap.nodes.radius = 1000
    n1_axon_soma2.nodes.radius = 500
    n1_overlap_context.nodes.radius = 250

    c1 = RL_colormap(n1_sidx)[0]
    c2 = RL_colormap(n2_sidx)[0]
    overlap_c = '#00FF2A'
    #colors = [c1, c1, c1, c2, c2, overlap_c]
    #navis.plot3d([n1_axon, n1_axon_soma, n1_axon_soma2, n2_dendrite, n2_dendrite_thick, n1_overlap, cns], radius=True, alpha=1,color=colors,     
     #   scatter_kws={'size':[30, 30], 'color':'magenta'},width=1000, height=1000)
    colors = ['gray', 'gray', 'orange']#overlap_c]
    colors = ['orange','orange', 'gray','black']
    #colors = [c2, c2, c1, 'black']
    navis.plot3d([n2_dendrite, n2_dendrite_thick, n1_axon, n1_overlap, cns], radius=True, alpha=1,color=colors,     
        scatter_kws={'size':[20, 20],'color':'cyan'},width=1500, height=1000)
        # magenta: '#FF3399'
    print('cable overlap is %f' % n1_overlap.cable_length)

def plot_error_diff_combined(MN_DF_all,min_err_trials=0):

    sns.set(rc={'figure.figsize':(1.5,3)}, font_scale=1.5, style='ticks')
    plt.subplots_adjust(hspace = .4, wspace=0.4)
    pal = {'pyramidal':'green','non pyramidal':'purple', 'unknown':'gray'}
    sig_pal = {True:'red',False:'grey'}

    MN_DF_all = MN_DF_all[MN_DF_all.n_trials_pref_error >= min_err_trials]
    MN_DF_all = MN_DF_all[MN_DF_all.n_trials_nonpref_error >= min_err_trials]

    MN_DF_all['err_diff_combined_sig'] = np.logical_or(MN_DF_all.err_diff_combined_prc > 97.5,MN_DF_all.err_diff_combined_prc < 2.5)
    #MN_DF_all['err_diff_combined_sig'] = MN_DF_all.err_diff_combined_prc > 95
    
    #g = sns.swarmplot(data = MN_DF_all, x='type',y='err_diff_combined', order = ['pyramidal','non pyramidal'], hue='err_diff_combined_sig', palette=sig_pal,color='k',s=2.5)
    #my_violinplot(data = MN_DF_all, x='type',y='err_diff_combined', order = ['pyramidal','non pyramidal'],palette = pal, cut=1, saturation =0.5) 

    g = sns.swarmplot(data = MN_DF_all, x='type',y='err_diff_combined', order = ['pyramidal','non pyramidal'], hue='type',palette = pal, dodge=False, s=2.5)
    my_violinplot(data = MN_DF_all, x='type',y='err_diff_combined', order = ['pyramidal','non pyramidal'], color='white', cut=1, saturation =0.5) 
    #kwargs={"marker":'^'}

    g.axhline(y=0,color='k', linewidth=1)
    g.get_legend().remove()
    #g.set_ylim([-1,5])
    g.set_ylabel('Err. Trial Sel. Degr. \n ($Δ_{pref} + Δ_{non}$)')
    g.set_xlabel('')
    g.set_xticklabels(['Exc','Inh'])

    MN_DF_E = MN_DF_all[MN_DF_all.type=='pyramidal'].dropna(subset=['err_diff_combined'])
    MN_DF_I = MN_DF_all[MN_DF_all.type=='non pyramidal'].dropna(subset=['err_diff_combined'])
    print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E), len(MN_DF_I)))
    (E_pref_med,I_pref_med)  = (np.median(MN_DF_E.err_diff_combined),np.median(MN_DF_I.err_diff_combined))
    (E_pref_avg,I_pref_avg)  = (np.mean(MN_DF_E.err_diff_combined),np.mean(MN_DF_I.err_diff_combined))

    print('Exc Pref: %f +- [%f,%f]' % (E_pref_med, np.percentile(MN_DF_E.err_diff_combined,25)-E_pref_med,np.percentile(MN_DF_E.err_diff_combined,75)-E_pref_med))
    print('Inh Pref: %f +- [%f,%f]' % (I_pref_med, np.percentile(MN_DF_I.err_diff_combined,25)-I_pref_med,np.percentile(MN_DF_I.err_diff_combined,75)-I_pref_med))
    print(stats.wilcoxon(MN_DF_all[MN_DF_all.type=='pyramidal'].err_diff_combined.values))
    print(stats.wilcoxon(MN_DF_all[MN_DF_all.type=='non pyramidal'].err_diff_combined.values))
    print(stats.mannwhitneyu(MN_DF_E.err_diff_combined.values, MN_DF_I.err_diff_combined.values))

    sns.despine()   

def plot_error_diff(MN_DF_all, min_err_trials=0, statistic = 'mean'):
    # Plot err diff plots (old metrics with pref/non-pref
    sns.set(rc={'figure.figsize':(4,3)}, font_scale=1.5, style='ticks')
    plt.subplots_adjust(hspace = .4, wspace=0.4)
    pal = {'pyramidal':'white','non pyramidal':'white'}
    sig_pal = {True:'red',False:'grey'}
    MN_DF_all['err_diff_pref_sig'] = np.logical_or(MN_DF_all.err_diff_pref_prc > 97.5,MN_DF_all.err_diff_pref_prc < 2.5)
    MN_DF_all['err_diff_nonpref_sig'] = np.logical_or(MN_DF_all.err_diff_nonpref_prc > 97.5,MN_DF_all.err_diff_nonpref_prc < 2.5)

    MN_DF_all.dropna(subset=['err_diff_nonpref','err_diff_pref'], inplace=True)
    plt.subplot(1,2,1)

    MN_DF_all_pref = MN_DF_all[MN_DF_all.n_trials_pref_error >= min_err_trials]
    g=my_violinplot(data = MN_DF_all_pref, x='type',y='err_diff_pref', order = ['pyramidal','non pyramidal'],palette = pal, cut=1, saturation =0.5) 
    #kwargs={"marker":'^'}
    sns.swarmplot(data = MN_DF_all_pref, x='type',y='err_diff_pref', order = ['pyramidal','non pyramidal'], hue='err_diff_pref_sig', linewidth=.5,palette=sig_pal,color='k',s=3)
    g.axhline(y=0,color='k', linewidth=1)
    g.get_legend().remove()
    #g.set_ylim([-1,5])
    g.set_ylabel('Δ Activity (err. - corr.)')
    g.set_xlabel('')
    g.set_xticklabels(['Exc','Inh'])
    g.set_title('pref.')

    plt.subplot(1,2,2)
    MN_DF_all_nonpref = MN_DF_all[MN_DF_all.n_trials_nonpref_error >= min_err_trials]
    g=my_violinplot(data = MN_DF_all_nonpref, x='type',y='err_diff_nonpref', order = ['pyramidal','non pyramidal'], palette = pal, cut=1, saturation = 0.5)
    sns.swarmplot(data = MN_DF_all_nonpref, x='type',y='err_diff_nonpref', order = ['pyramidal','non pyramidal'], hue='err_diff_nonpref_sig', linewidth=.5,palette=sig_pal,color='k',s=3)
    g.axhline(y=0,color='k', linewidth=1)
    g.get_legend().remove()
    #g.set_ylim([-1,5])
    g.set_ylabel('')
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(['Exc','Inh'])
    g.set_title('non-pref.')
    sns.despine()

    MN_DF_E = MN_DF_all[MN_DF_all.type=='pyramidal'].dropna(subset=['err_diff_pref','err_diff_nonpref'])
    MN_DF_I = MN_DF_all[MN_DF_all.type=='non pyramidal'].dropna(subset=['err_diff_pref','err_diff_nonpref'])
    print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E), len(MN_DF_I)))
    
    print('Preferred')
    MN_DF_E_pref = MN_DF_all_pref[MN_DF_all_pref.type=='pyramidal'].dropna(subset=['err_diff_pref'])
    MN_DF_I_pref = MN_DF_all_pref[MN_DF_all_pref.type=='non pyramidal'].dropna(subset=['err_diff_pref'])
    print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E_pref), len(MN_DF_I_pref)))
    if statistic == 'mean':
        (E_pref,I_pref)  = (np.mean(MN_DF_E_pref.err_diff_pref),np.mean(MN_DF_I_pref.err_diff_pref))
    elif statistic == 'median':
        (E_pref,I_pref)  = (np.median(MN_DF_E_pref.err_diff_pref),np.median(MN_DF_I_pref.err_diff_pref))
    print('Exc Nonpref: %f +- [%f,%f]' % (E_pref, np.percentile(MN_DF_E_pref.err_diff_pref,25)-E_pref,np.percentile(MN_DF_E_pref.err_diff_pref,75)-E_pref))
    print('Inh Nonpref: %f +- [%f,%f]' % (I_pref, np.percentile(MN_DF_I_pref.err_diff_pref,25)-I_pref,np.percentile(MN_DF_I_pref.err_diff_pref,75)-I_pref))
    print(stats.wilcoxon(MN_DF_E_pref.err_diff_pref.values))
    print(stats.wilcoxon(MN_DF_I_pref.err_diff_pref.values))
    print(stats.mannwhitneyu(MN_DF_E_pref.err_diff_pref.values, MN_DF_I_pref.err_diff_pref.values))

    print('Non Preferred')
    MN_DF_E_nonpref = MN_DF_all_nonpref[MN_DF_all_nonpref.type=='pyramidal'].dropna(subset=['err_diff_nonpref'])
    MN_DF_I_nonpref = MN_DF_all_nonpref[MN_DF_all_nonpref.type=='non pyramidal'].dropna(subset=['err_diff_nonpref'])
    print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E_nonpref), len(MN_DF_I_nonpref)))
    if statistic == 'mean':
        (E_nonpref,I_nonpref)  = (np.mean(MN_DF_E_nonpref.err_diff_nonpref),np.mean(MN_DF_I_nonpref.err_diff_nonpref))
    elif statistic == 'median':
        (E_nonpref,I_nonpref)  = (np.median(MN_DF_E_nonpref.err_diff_nonpref),np.median(MN_DF_I_nonpref.err_diff_nonpref))
    print('Exc Nonpref: %f +- [%f,%f]' % (E_nonpref, np.percentile(MN_DF_E_nonpref.err_diff_nonpref,25)-E_nonpref,np.percentile(MN_DF_E_nonpref.err_diff_nonpref,75)-E_nonpref))
    print('Inh Nonpref: %f +- [%f,%f]' % (I_nonpref, np.percentile(MN_DF_I_nonpref.err_diff_nonpref,25)-I_nonpref,np.percentile(MN_DF_I_nonpref.err_diff_nonpref,75)-I_nonpref))
    
    print(stats.wilcoxon(MN_DF_E_nonpref.err_diff_nonpref.values))
    print(stats.wilcoxon(MN_DF_I_nonpref.err_diff_nonpref.values))
    print(stats.mannwhitneyu(MN_DF_E_nonpref.err_diff_nonpref.values, MN_DF_I_nonpref.err_diff_nonpref.values))

def plot_error_diff_compare(MN_DF_all, min_err_trials=0, statistic = 'mean'):
    # Plot err diff plots (old metrics with pref/non-pref
    sns.set(rc={'figure.figsize':(4,3)}, font_scale=1.5, style='ticks')
    plt.subplots_adjust(hspace = .4, wspace=0.4)
    pal = {'pyramidal':'white','non pyramidal':'white'}
    sig_pal = {True:'red',False:'grey'}
    MN_DF_all['err_diff_pref_sig'] = np.logical_or(MN_DF_all.err_diff_pref_prc > 97.5,MN_DF_all.err_diff_pref_prc < 2.5)
    MN_DF_all['err_diff_nonpref_sig'] = np.logical_or(MN_DF_all.err_diff_nonpref_prc > 97.5,MN_DF_all.err_diff_nonpref_prc < 2.5)

    MN_DF_all.dropna(subset=['err_diff_nonpref','err_diff_pref'], inplace=True)
    MN_DF_all_pref = MN_DF_all[MN_DF_all.n_trials_pref_error >= min_err_trials]
    MN_DF_all_pref['err_diff'] = MN_DF_all_pref.err_diff_pref.values
    MN_DF_all_pref['trial_type'] = 'pref'
    MN_DF_all_nonpref = MN_DF_all[MN_DF_all.n_trials_nonpref_error >= min_err_trials]
    MN_DF_all_nonpref['err_diff'] = MN_DF_all_nonpref.err_diff_nonpref.values
    MN_DF_all_nonpref['trial_type'] = 'non_pref'
    MN_DF_combined = pd.concat([MN_DF_all_nonpref, MN_DF_all_pref])

    # Plot excitatory
    plt.subplot(1,2,1)
    MN_DF_all_E = MN_DF_combined[MN_DF_combined.type=='pyramidal']
    g=my_violinplot(data = MN_DF_all_E, x='trial_type',y='err_diff', order = ['non_pref','pref'],color='w', scale='width', cut=2, saturation =0.5) 
    #kwargs={"marker":'^'}
    sns.swarmplot(data = MN_DF_all_E, x='trial_type',y='err_diff', order = ['non_pref','pref'], linewidth=0,color='green',s=2)
    g.axhline(y=0,color='k', linestyle = '--',linewidth=1.5)
    #g.get_legend().remove()
    g.set_ylim([-1,5])
    g.set_ylabel('Δ Activity (err. - corr.)')
    g.set_xlabel('')
    g.set_xticklabels(['non-pref.','preferred'], rotation = 45)
    #g.set_title('Excitatory')

    plt.subplot(1,2,2)
    MN_DF_all_I = MN_DF_combined[MN_DF_combined.type=='non pyramidal']
    g=my_violinplot(data = MN_DF_all_I, x='trial_type',y='err_diff', order = ['non_pref','pref'], color='w', scale='width', cut=2, saturation = 0.5)
    sns.swarmplot(data = MN_DF_all_I, x='trial_type',y='err_diff', order = ['non_pref','pref'], linewidth=0,color='purple',s=3)
    g.axhline(y=0,color='k', linestyle = '--',linewidth=1.5)
    #g.get_legend().remove()
    g.set_ylim([-1,5])
    g.set_ylabel('')
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(['non-pref.','preferred'], rotation = 45)
    #g.set_title('Inhibitory')
    sns.despine()

    MN_DF_E = MN_DF_all[MN_DF_all.type=='pyramidal'].dropna(subset=['err_diff_pref','err_diff_nonpref'])
    MN_DF_I = MN_DF_all[MN_DF_all.type=='non pyramidal'].dropna(subset=['err_diff_pref','err_diff_nonpref'])
    print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E), len(MN_DF_I)))
    
    print('Preferred')
    MN_DF_E_pref = MN_DF_all_pref[MN_DF_all_pref.type=='pyramidal'].dropna(subset=['err_diff_pref'])
    MN_DF_I_pref = MN_DF_all_pref[MN_DF_all_pref.type=='non pyramidal'].dropna(subset=['err_diff_pref'])
    print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E_pref), len(MN_DF_I_pref)))
    if statistic == 'mean':
        (E_pref,I_pref)  = (np.mean(MN_DF_E_pref.err_diff_pref),np.mean(MN_DF_I_pref.err_diff_pref))
    elif statistic == 'median':
        (E_pref,I_pref)  = (np.median(MN_DF_E_pref.err_diff_pref),np.median(MN_DF_I_pref.err_diff_pref))
    print('Exc Nonpref: %f +- [%f,%f]' % (E_pref, np.percentile(MN_DF_E_pref.err_diff_pref,25)-E_pref,np.percentile(MN_DF_E_pref.err_diff_pref,75)-E_pref))
    print('Inh Nonpref: %f +- [%f,%f]' % (I_pref, np.percentile(MN_DF_I_pref.err_diff_pref,25)-I_pref,np.percentile(MN_DF_I_pref.err_diff_pref,75)-I_pref))
    print(stats.wilcoxon(MN_DF_E_pref.err_diff_pref.values))
    print(stats.wilcoxon(MN_DF_I_pref.err_diff_pref.values))
    print(stats.mannwhitneyu(MN_DF_E_pref.err_diff_pref.values, MN_DF_I_pref.err_diff_pref.values))

    print('Non Preferred')
    MN_DF_E_nonpref = MN_DF_all_nonpref[MN_DF_all_nonpref.type=='pyramidal'].dropna(subset=['err_diff_nonpref'])
    MN_DF_I_nonpref = MN_DF_all_nonpref[MN_DF_all_nonpref.type=='non pyramidal'].dropna(subset=['err_diff_nonpref'])
    print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E_nonpref), len(MN_DF_I_nonpref)))
    if statistic == 'mean':
        (E_nonpref,I_nonpref)  = (np.mean(MN_DF_E_nonpref.err_diff_nonpref),np.mean(MN_DF_I_nonpref.err_diff_nonpref))
    elif statistic == 'median':
        (E_nonpref,I_nonpref)  = (np.median(MN_DF_E_nonpref.err_diff_nonpref),np.median(MN_DF_I_nonpref.err_diff_nonpref))
    print('Exc Nonpref: %f +- [%f,%f]' % (E_nonpref, np.percentile(MN_DF_E_nonpref.err_diff_nonpref,25)-E_nonpref,np.percentile(MN_DF_E_nonpref.err_diff_nonpref,75)-E_nonpref))
    print('Inh Nonpref: %f +- [%f,%f]' % (I_nonpref, np.percentile(MN_DF_I_nonpref.err_diff_nonpref,25)-I_nonpref,np.percentile(MN_DF_I_nonpref.err_diff_nonpref,75)-I_nonpref))
    
    print(stats.wilcoxon(MN_DF_E_nonpref.err_diff_nonpref.values))
    print(stats.wilcoxon(MN_DF_I_nonpref.err_diff_nonpref.values))
    print(stats.mannwhitneyu(MN_DF_E_nonpref.err_diff_nonpref.values, MN_DF_I_nonpref.err_diff_nonpref.values))

def plot_err_degr(MN_DF_all, min_err_trials = 3, statistic = 'median', hue = None, order = None, make_plots = True):

    def calc_err_degr_stats(MN_DF_all):
        # Calc statistics for error trial selectivity degradation 
        err_degr = pd.DataFrame([])
        err_degr_shuf = pd.DataFrame([])
        err_degr_boot = pd.DataFrame([])

        MN_DF_all_pref = MN_DF_all[MN_DF_all.n_trials_pref_error >= min_err_trials]    
        MN_DF_all_nonpref = MN_DF_all[MN_DF_all.n_trials_nonpref_error >= min_err_trials]

        MN_DF_E = MN_DF_all[MN_DF_all.type=='pyramidal'].dropna(subset=['err_diff_pref','err_diff_nonpref'])
        MN_DF_I = MN_DF_all[MN_DF_all.type=='non pyramidal'].dropna(subset=['err_diff_pref','err_diff_nonpref'])
        print('Total Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E), len(MN_DF_I)))

        print('Num Trials: %i to %i' % (np.min(MN_DF_all.n_trials),np.max(MN_DF_all.n_trials))) 

        print('Preferred')
        MN_DF_E_pref = MN_DF_all_pref[MN_DF_all_pref.type=='pyramidal'].dropna(subset=['err_diff_pref'])
        MN_DF_I_pref = MN_DF_all_pref[MN_DF_all_pref.type=='non pyramidal'].dropna(subset=['err_diff_pref'])
        print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E_pref), len(MN_DF_I_pref)))
        if statistic == 'mean':
            (E_pref,I_pref)  = (np.mean(MN_DF_E_pref.err_diff_pref),np.mean(MN_DF_I_pref.err_diff_pref))
        elif statistic == 'median':
            (E_pref,I_pref)  = (np.median(MN_DF_E_pref.err_diff_pref),np.median(MN_DF_I_pref.err_diff_pref))
        #print('Exc Pref: %f +- [%f,%f]' % (E_pref, np.percentile(MN_DF_E_pref.err_diff_pref,25)-E_pref,np.percentile(MN_DF_E_pref.err_diff_pref,75)-E_pref))
        #print('Inh Pref: %f +- [%f,%f]' % (I_pref, np.percentile(MN_DF_I_pref.err_diff_pref,25)-I_pref,np.percentile(MN_DF_I_pref.err_diff_pref,75)-I_pref))
        #print(stats.wilcoxon(MN_DF_E_pref.err_diff_pref.values))
        #print(stats.wilcoxon(MN_DF_I_pref.err_diff_pref.values))
        #print(stats.mannwhitneyu(MN_DF_E_pref.err_diff_pref.values, MN_DF_I_pref.err_diff_pref.values))

        print('Non Preferred')
        MN_DF_E_nonpref = MN_DF_all_nonpref[MN_DF_all_nonpref.type=='pyramidal'].dropna(subset=['err_diff_nonpref'])
        MN_DF_I_nonpref = MN_DF_all_nonpref[MN_DF_all_nonpref.type=='non pyramidal'].dropna(subset=['err_diff_nonpref'])
        print('Excitatory: n = %i, Inh: n = %i' % (len(MN_DF_E_nonpref), len(MN_DF_I_nonpref)))
        if statistic == 'mean':
            (E_nonpref,I_nonpref)  = (np.mean(MN_DF_E_nonpref.err_diff_nonpref),np.mean(MN_DF_I_nonpref.err_diff_nonpref))
        elif statistic == 'median':
            (E_nonpref,I_nonpref)  = (np.median(MN_DF_E_nonpref.err_diff_nonpref),np.median(MN_DF_I_nonpref.err_diff_nonpref))
        #print('Exc Nonpref: %f +- [%f,%f]' % (E_nonpref, np.percentile(MN_DF_E_nonpref.err_diff_nonpref,25)-E_nonpref,np.percentile(MN_DF_E_nonpref.err_diff_nonpref,75)-E_nonpref))
        #print('Inh Nonpref: %f +- [%f,%f]' % (I_nonpref, np.percentile(MN_DF_I_nonpref.err_diff_nonpref,25)-I_nonpref,np.percentile(MN_DF_I_nonpref.err_diff_nonpref,75)-I_nonpref))

        err_diff_E =  E_nonpref - E_pref
        err_diff_I = I_nonpref - I_pref
        err_diff_EvI = err_diff_E - err_diff_I

        err_degr = pd.DataFrame({'err_degr': [err_diff_E, err_diff_I], 'type': ['excitatory','inhibitory']})
        #err_degr['err_degr_E']  = E_nonpref - E_pref
        #err_degr['err_degr_I']  = I_nonpref - I_pref
        print('Error trial sel. degr.: E: %0.3f, I:%0.3f' %(err_diff_E, err_diff_I))

        # Calc p-value compared to shuffle (shuf error trial identity)
        n_shuf = MN_DF_all.n_shuf.values[0]
        err_diff_E_shufs = np.empty((n_shuf))
        err_diff_I_shufs = np.empty((n_shuf))
        err_diff_EvI_shufs = np.empty((n_shuf))

        for shuf_idx in np.arange(n_shuf):
            if statistic == 'median':
                err_diff_E_shufs[shuf_idx] = np.nanmedian([i[shuf_idx] for i in MN_DF_E_nonpref.err_diff_nonpref_shuf]) - np.nanmedian([i[shuf_idx] for i in        MN_DF_E_pref.err_diff_pref_shuf])
                err_diff_I_shufs[shuf_idx] = np.nanmedian([i[shuf_idx] for i in MN_DF_I_nonpref.err_diff_nonpref_shuf]) - np.nanmedian([i[shuf_idx] for i in MN_DF_I_pref.err_diff_pref_shuf])
            elif statistic == 'mean':
                err_diff_E_shufs[shuf_idx] = np.nanmean([i[shuf_idx] for i in MN_DF_E_nonpref.err_diff_nonpref_shuf]) - np.nanmean([i[shuf_idx] for i in        MN_DF_E_pref.err_diff_pref_shuf])
                err_diff_I_shufs[shuf_idx] = np.nanmean([i[shuf_idx] for i in MN_DF_I_nonpref.err_diff_nonpref_shuf]) - np.nanmean([i[shuf_idx] for i in MN_DF_I_pref.err_diff_pref_shuf])
            err_diff_EvI_shufs[shuf_idx] = err_diff_E_shufs[shuf_idx] - err_diff_I_shufs[shuf_idx]
        err_degr_shuf = pd.concat([pd.DataFrame({'err_degr': err_diff_E_shufs, 'type': 'excitatory'}),pd.DataFrame({'err_degr': err_diff_I_shufs, 'type': 'inhibitory'})])
        #err_degr_shuf['err_degr_E'] =  err_diff_E_shufs
        #err_degr_shuf['err_degr_I'] = err_diff_I_shufs

        #Two-tailed test
        perc_E_1 = stats.percentileofscore(err_diff_E_shufs, err_diff_E)
        p_E_1 = min(perc_E_1/100, (100-perc_E_1)/100)
        perc_E_2 = stats.percentileofscore(err_diff_E_shufs, -err_diff_E)
        p_E_2 = min(perc_E_2/100, (100-perc_E_2)/100)
        p_E = p_E_1+p_E_2

        perc_I_1 = stats.percentileofscore(err_diff_I_shufs, err_diff_I)
        p_I_1 = min(perc_I_1/100, (100-perc_I_1)/100)
        perc_I_2 = stats.percentileofscore(err_diff_I_shufs, -err_diff_I)
        p_I_2 = min(perc_I_1/100, (100-perc_I_2)/100)
        p_I = p_I_1 + p_I_2

        perc_diff_1 = stats.percentileofscore(err_diff_EvI_shufs, err_diff_EvI)
        p_diff_1 = min(perc_diff_1/100, (100-perc_diff_1)/100)
        perc_diff_2 = stats.percentileofscore(-err_diff_EvI_shufs, err_diff_EvI)
        p_diff_2 = min(perc_diff_2/100, (100-perc_diff_2)/100)
        p_diff = p_diff_1+p_diff_2

        print('p-values for permutation test: E: %0.3f I: %0.3f, Diff: %0.3f' %(p_E, p_I, p_diff))
        #sns.violinplot((err_diff_E_shufs, err_diff_I_shufs))

        # Make bootstrap dist for plotting (bootstrap neurons)
        err_diff_E_boot = np.empty((n_shuf))
        err_diff_I_boot = np.empty((n_shuf))

        for shuf_idx in np.arange(n_shuf):
            if statistic == 'median':
                err_diff_E_boot[shuf_idx] = np.nanmedian(MN_DF_E_nonpref.sample(frac=1,replace=True).err_diff_nonpref.values) - np.nanmedian(MN_DF_E_pref.sample(frac=1,replace=True).err_diff_pref.values)
                err_diff_I_boot[shuf_idx] = np.nanmedian(MN_DF_I_nonpref.sample(frac=1,replace=True).err_diff_nonpref.values) - np.nanmedian(MN_DF_I_pref.sample(frac=1,replace=True).err_diff_pref.values)
            elif statistic == 'mean':
                err_diff_E_boot[shuf_idx] = np.nanmean(MN_DF_E_nonpref.sample(frac=1,replace=True).err_diff_nonpref.values) - np.nanmean(MN_DF_E_pref.sample(frac=1,replace=True).err_diff_pref.values)
                err_diff_I_boot[shuf_idx] = np.nanmean(MN_DF_I_nonpref.sample(frac=1,replace=True).err_diff_nonpref.values) - np.nanmean(MN_DF_I_pref.sample(frac=1,replace=True).err_diff_pref.values)

        err_degr_boot = pd.concat([pd.DataFrame({'err_degr': err_diff_E_boot, 'type': 'excitatory'}),pd.DataFrame({'err_degr': err_diff_I_boot, 'type': 'inhibitory'})])
        return (err_degr, err_degr_boot)

    if hue is None:
        (err_degr, err_degr_boot) = calc_err_degr_stats(MN_DF_all)
        (err_degr_all, err_degr_boot_all) = (err_degr, err_degr_boot) 
        if make_plots:
            sns.barplot(data=err_degr_boot, x='type',y='err_degr', alpha=0, errorbar=('pi', 95), capsize=0.1)
            sns.barplot(data=err_degr, x='type',y='err_degr')
    else:
        err_degr_all = pd.DataFrame()
        err_degr_boot_all = pd.DataFrame()
        for hue_val in np.unique(MN_DF_all[hue].values):
            print(hue_val)
            (err_degr, err_degr_boot) = calc_err_degr_stats(MN_DF_all[MN_DF_all[hue] == hue_val])
            err_degr[hue] = hue_val
            err_degr_boot[hue] = hue_val
            err_degr_all = pd.concat([err_degr_all, err_degr])
            err_degr_boot_all = pd.concat([err_degr_boot_all, err_degr_boot])
        if make_plots:
            sns.barplot(data=err_degr_boot_all, x=hue, y='err_degr', hue='type', order = order, alpha=0, errorbar=('pi', 95), capsize=0.1)
            sns.barplot(data=err_degr_all, x=hue,y='err_degr', hue='type', order = order)

    return (err_degr_all, err_degr_boot_all)