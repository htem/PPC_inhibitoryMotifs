from all_imports import *
import warnings
from Compute_noise_corr import*
warnings.filterwarnings('ignore', category=RuntimeWarning)
from scipy.stats import wilcoxon, ttest_1samp

def wilcoxon_axis1(A):
    p = empty(A.shape[0])
    for i in range(A.shape[0]):
        try:
            x = ttest_1samp(A[i], 0, nan_policy='omit')[1]
            # x = wilcoxon(A[i])[1]
        except:
            x = 1
        else:
            x = ttest_1samp(A[i], 0, nan_policy='omit')[1]
            # x = wilcoxon(A[i])[1]
        p[i] = x
    return p

def compute_global_act(A, session, pair=[]):
    activity_combined = []
    l=0
    for i in range(143):
        if i not in pair:
            if A[i,session].shape[1] != 0:
                activity_combined.append(A[i,session][times_used])
                l+=1
    return sum(array(activity_combined),axis=0)/l



# 0 time
# 1 x position
# 2 y position
# 3 heading angle
# 4 x velocity
# 5 y velocity "lateral running speed?"
# 6 vr.cuePos
# 7 vr.isReward
# 8 vr.inITI
# 9 vr.greyFac (always .5)

Data_all = loadmat('230621_single_trial_data.mat')
activity = Data_all['neuronAct']
cues = Data_all['allCue']
corrects = Data_all['allCorrect']
behavior = loadmat('behavior_sess.mat')['behavior_sess']
connections = loadmat('connections.mat')
there_is_session = loadmat('there_is_session.mat')['there_is_session'].astype(bool)
source_EI = connections['source_EI'][0] - 1
target_EI = connections['target_EI'][0] - 1
synden_EI = connections['synden_EI'][0]
source_IE = connections['source_IE'][0] - 1
target_IE = connections['target_IE'][0] - 1
synden_IE = connections['synden_IE'][0]

times_used = range(13,76)

n_EI = len(target_EI)
n_IE = len(target_IE)
noisecorr_EI = empty(n_EI)
noisecorr_IE = empty(n_IE)

##
filter_time = True
remove_confounder = True
behav_vars = [1,2,3,4,5] # 1,2,3,4,5

if remove_confounder:
    rem = 'rem'
    which_var = '_'.join(str(item) for item in behav_vars)
    # which_var += ',global'
    # which_var += ',except'

    # which_var = 'global'
    # which_var = 'except'
else:
    rem = ''
    which_var =''



type_corr, how = 'Avg_after', 'Combined'
thrs = 0.2


for i in range(n_EI):
    # print(f'i_IE={i}')
    i_source = source_EI[i]
    i_target = target_EI[i]
    noise_corr_sessions = []
    list_sessions = get_common_sessions(i_source, i_target, there_is_session)
    for i_sess in list_sessions:
        Act_source_sess = activity[i_source, i_sess][None][:, times_used, :]
        Act_target_sess = activity[i_target, i_sess][None][:, times_used, :]
        behavior_sess = transpose(behavior[0, i_sess], (0, 2, 1))
        behavior_z_sess = my_z_score(behavior_sess)
        behavior_z_sess = behavior_z_sess[behav_vars][:, times_used, :]
        global_activity_all = compute_global_act(activity, i_sess, pair=[])[None]
        global_activity_except = compute_global_act(activity, i_sess, pair=[i_source, i_target])[None]

        if filter_time:
            filter_t = (wilcoxon_axis1(Act_source_sess[0])<0.05) & (wilcoxon_axis1(Act_target_sess[0])<0.05)
        else:
            filter_t = range(len(times_used))

        Act_source_sess = Act_source_sess[:,filter_t,:]
        Act_target_sess = Act_target_sess[:,filter_t,:]
        behavior_z_sess = behavior_z_sess[:,filter_t,:]
        global_activity_all = global_activity_all[:,filter_t,:]
        global_activity_except = global_activity_except[:,filter_t,:]
        print(i, i_sess, sum(filter_t))
        cues_sess = cues[0, i_sess][0]

        if remove_confounder:
            Regressors = behavior_z_sess
            # Regressors = global_activity_all
            # Regressors = global_activity_except
            # Regressors = vstack((behavior_z_sess, global_activity_all))
            # Regressors = vstack((behavior_z_sess, global_activity_except))

            Act_source_sess = remove_confounder_separate_cues(Act_source_sess, Regressors, cues_sess)
            Act_target_sess = remove_confounder_separate_cues(Act_target_sess, Regressors, cues_sess)

        activity_sess = vstack((Act_source_sess, Act_target_sess))
        rho = compute_noise_corr(0, 1, activity_sess, corrects, cues_sess, 0., type_corr, how)
        noise_corr_sessions.append(rho)

    noisecorr_EI[i] = nanmean(noise_corr_sessions)


for i in range(n_IE):
    print(f'i_IE={i}')
    i_source = source_IE[i]
    i_target = target_IE[i]
    noise_corr_sessions = []
    list_sessions = get_common_sessions(i_source, i_target, there_is_session)
    for i_sess in list_sessions:
        Act_source_sess = activity[i_source, i_sess][None][:, times_used, :]
        Act_target_sess = activity[i_target, i_sess][None][:, times_used, :]
        behavior_sess = transpose(behavior[0, i_sess], (0, 2, 1))
        behavior_z_sess = my_z_score(behavior_sess)
        behavior_z_sess = behavior_z_sess[behav_vars][:, times_used, :]
        global_activity_all = compute_global_act(activity, i_sess, pair=[])[None]
        global_activity_except = compute_global_act(activity, i_sess, pair=[i_source, i_target])[None]
        cues_sess = cues[0, i_sess][0]

        if filter_time:
            filter_t = (wilcoxon_axis1(Act_source_sess[0]) < 0.05) & (wilcoxon_axis1(Act_target_sess[0]) < 0.05)
        else:
            filter_t = range(len(times_used))

        Act_source_sess = Act_source_sess[:,filter_t,:]
        Act_target_sess = Act_target_sess[:,filter_t,:]
        behavior_z_sess = behavior_z_sess[:,filter_t,:]
        global_activity_all = global_activity_all[:, filter_t, :]
        global_activity_except = global_activity_except[:, filter_t, :]

        if remove_confounder:
            Regressors = behavior_z_sess
            # Regressors = global_activity_all
            # Regressors = global_activity_except
            # Regressors = vstack((behavior_z_sess, global_activity_all))
            # Regressors = vstack((behavior_z_sess, global_activity_except))

            Act_source_sess = remove_confounder_separate_cues(Act_source_sess, Regressors, cues_sess)
            Act_target_sess = remove_confounder_separate_cues(Act_target_sess, Regressors, cues_sess)

        activity_sess = vstack((Act_source_sess, Act_target_sess))
        rho = compute_noise_corr(0, 1, activity_sess, corrects, cues_sess, 0., type_corr, how)
        noise_corr_sessions.append(rho)

    noisecorr_IE[i] = nanmean(noise_corr_sessions)


##
fig = figure(figsize=(9,3))

ax = fig.add_subplot(1,3,1)
x,y = noisecorr_IE, synden_IE
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'E-to-I {type_corr}-{how}{which_var}/bySess,\n c={c:.2f}, p={p:.3f}', fontsize=8)

ax = fig.add_subplot(1,3,2)
x,y = noisecorr_EI, synden_EI
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'I-to-E {type_corr}-{how}{which_var}/bySess,\n c={c:.2f}, p={p:.3f}', fontsize=8)

ax = fig.add_subplot(1,3,3)
x = concatenate((noisecorr_IE-nanmean(noisecorr_IE), noisecorr_EI-nanmean(noisecorr_EI)))
y = concatenate((synden_IE-nanmean(synden_IE), -synden_EI+nanmean(synden_EI)))
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'Pooled {type_corr}-{how}\n{which_var}/bySess,\n c={c:.2f}, p={p:.3f}', fontsize=8)

tight_layout()
##

# savefig(f'/Users/giuliobondanelli/Desktop/noise_corr_plots/noisecorr_{type_corr}_{how}_{which_var}_bySess.pdf')


##

data = {'noisecorr_IE':noisecorr_IE,
        'noisecorr_EI':noisecorr_EI,
        'synden_IE':synden_IE,
        'synden_EI': synden_EI}

save(f'/Users/giuliobondanelli/OneDrive - Fondazione Istituto Italiano Tecnologia/\
Code/aaron_matlab/noise_corr_data/noisecorr_{type_corr}_{how}_{rem}_{which_var}_bySess.npy', data)
##


##

