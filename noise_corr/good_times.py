from all_imports import *
import warnings
from Compute_noise_corr import*
warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.stats import wilcoxon,ttest_1samp

activity = loadmat('Act_combined_GB.mat')['Act_combined_GB']
activity = activity[:,range(13,76),:]
ncells = activity.shape[0]

def wilcoxon_axis1(A):
    p = empty(A.shape[0])
    for i in range(A.shape[0]):
        try:
            x = ttest_1samp(A[i], 0, nan_policy='omit')[1]
        except:
            x = 1
            print('error')
        else:
            x = ttest_1samp(A[i], 0, nan_policy='omit')[1]
        p[i] = x
    return p

def get_good_times(i, j, activity):
    Ai = activity[i]
    Aj = activity[j]
    pi = wilcoxon_axis1(Ai)
    pj = wilcoxon_axis1(Aj)
    return (pi<0.05) & (pj<0.05)

# def get_good_times(i, j, activity):
#     Ai = activity[i]
#     Aj = activity[j]
#     pi = nanmean(Ai,axis=1)
#     pj = nanmean(Aj,axis=1)
#     return (pi>0.05) & (pj>0.05)

