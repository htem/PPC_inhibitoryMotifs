from matplotlib.pyplot import *
from matplotlib.colors import LinearSegmentedColormap
from numpy import *
import rc_parameters

def my_figure(figsize = (2,2)):
    fig, ax = subplots(figsize = figsize, dpi=300)
    fmt = matplotlib.ticker.StrMethodFormatter("{x}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    return fig,ax

def my_fill_between(x, F, col, colfill, labels,  **pars):
    ls = pars['ls']
    lw = pars['lw']
    ms = pars['markersize']
    m  = [nanmean(f,0) for f in F]
    s  = [nanstd(f,0) for f in F]
    ntrials = F[0].shape[0]
    a = sqrt(ntrials) if pars['err'] == 'se' else 1
    for i in range(len(m)):
        plot(x, m[i], ls, lw = lw, color = col[i], label = labels[i],markersize=ms)
        fill_between(x, m[i]-s[i]/a, m[i]+s[i]/a,color = colfill[i])

def my_boxplot(figsize, data, labels, rotation, facecolor, colorwhisk, colorcaps, colorfliers, width):
    fig = figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    flierprops      = dict(marker = 'o', markerfacecolor = colorfliers, markersize = 3, markeredgewidth = 0, linestyle = 'none')
    boxprops        = dict(linewidth = 1.)
    capprops        = dict(linewidth = 1.)
    bp = ax.boxplot(data, flierprops = flierprops, widths = width, patch_artist = True,
                     boxprops = boxprops, capprops = capprops)
    for i,box in enumerate(bp['boxes']):
        box.set(facecolor=facecolor[i], color=facecolor[i])
    for i,whisker in enumerate(bp['whiskers']):
        whisker.set(color = colorwhisk[i])
    for i, cap in enumerate(bp['caps']):
        cap.set(color = colorcaps[i])
    for median in bp['medians']:
        median.set(color='grey', linewidth=.8)
    xticks(arange(1,len(labels)+1,1), labels, rotation=rotation, fontsize=10)
    tight_layout()
    return ax

def define_colormap(colors, N):
    cm = LinearSegmentedColormap.from_list('new_cm', colors, N)
    return cm

def format_axes(ax, nfloatsx=None, nfloatsy=None, cbar = None, nfloatsz=None):
    fmt = matplotlib.ticker.StrMethodFormatter("{x}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    if nfloatsx is not None:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.' + str(nfloatsx) + 'f'))
    if nfloatsy is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.'+str(nfloatsy)+'f'))
    if cbar is not None:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.' + str(nfloatsz) + 'f'))

def plot_data_points(x, data, color='k', jitter=0, markersize=2):
    # data is n_observations x n_variables
    n_observations = data.shape[0]
    n_variables = data.shape[1]
    for i in range(n_observations):
        plot(x+random.normal(0,jitter,len(x)), data[i], 'o', color=color, markersize=markersize)
        
def my_violin(data, width, color_bodies, xTicks, color_points = None):
    positions = arange(1,len(data)+1)
    widths = [width] * len(data)
    data = [array(p) for p in data]
    data2 = []
    for D in data:
        data2.append(D[~isnan(D)])
    data = data2
    parts = violinplot(data, positions=positions, widths=widths, showmeans=True, showmedians=False, showextrema=False)
    i=0
    for pc in parts['bodies']:
        pc.set_facecolor(color_bodies[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
        pc.set_alpha(1)
        i+=1
    parts['cmeans'].set_color('k')
    parts['cmeans'].set_linewidth(1)

    # parts['cmedians'].set_color('k')
    # parts['cmedians'].set_linewidth(1)

    if color_points is not None:
        for i in range(len(data)):
            for j in range(len(data[i])):
                plot([positions[i] + random.normal(0, 0.15)], [data[i][j]], '.', markersize=1, alpha=1,
                     color=color_points[i])
    xticks(positions,xTicks)
