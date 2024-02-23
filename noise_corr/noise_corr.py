from all_imports import *
import warnings
from Compute_noise_corr import*
warnings.filterwarnings('ignore', category=RuntimeWarning)
from good_times import *

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

activity = loadmat('Act_combined_GB.mat')['Act_combined_GB']
behavior = loadmat('Behavior_combined_GB.mat')['Behav_combined_GB']
corrects = loadmat('Act_combined_GB.mat')['Corrects_combined_GB'][0]
cues = loadmat('Act_combined_GB.mat')['Cue_combined_GB'][0]
max_times_IE = loadmat('max_times_EtoI.mat')['max_times_EtoI'][0] - 1
max_times_EI = loadmat('max_times_ItoE.mat')['max_times_ItoE'][0] - 1
connections = loadmat('connections.mat')
source_EI = connections['source_EI'][0] - 1
target_EI = connections['target_EI'][0] - 1
synden_EI = connections['synden_EI'][0]
source_IE = connections['source_IE'][0] - 1
target_IE = connections['target_IE'][0] - 1
synden_IE = connections['synden_IE'][0]


times_used = range(13,76)
activity = activity[:,times_used,:]
behavior = behavior[:,times_used,:]

##
n_EI = len(target_EI)
n_IE = len(target_IE)
noisecorr_EI = empty(n_EI)
noisecorr_IE = empty(n_IE)

filter_time = True #take only times where there is significant activity for both source and target
remove_confounder = True #regress behavior or global activity; regressors used are defined elow in the two for loops
behav_vars = [1,2,3,4,5] # 1,2,3,4,5 #which behav variables to regress

if remove_confounder: #just strings
    rem = 'rem'
    which_var = '_'.join(str(item) for item in behav_vars)
    # which_var += ',global'
    # which_var += ',except'

    # which_var = 'global'
    # which_var = 'except'
else:
    rem = ''
    which_var =''

#type corr: 'Avg', 'Max_after', 'Avg_after'
# 'Avg': corr of activity avg over time
# 'Max_after': corr computed at each time, then take timepoint where |corr| is max and take the value of corr (signed)
# 'Avg_after': corr computed at each time, then take corr avg over time

# how: 'Combined', 'Separate'
# 'Combined': combine the trials for the two cues and compute corr
# 'Separate': compute corr for the two cues separately and take mean over cues

type_corr, how = 'Avg', 'Combined'
threshold = 0.0 #not used


for i in range(n_EI):
    print(f'i_EI={i}')
    i_source = source_EI[i]
    i_target = target_EI[i]

    if filter_time:
        filter_t = get_good_times(i_source, i_target, activity)
        if filter_time == 'Max':
            if isnan(max_times_EI[i]):
                continue
            else:
                filter_t = [int(max_times_EI[i])]
    else:
        filter_t = range(activity.shape[1])

    if remove_confounder:

        Act_source = activity[i_source, filter_t, :][None]
        Act_target = activity[i_target, filter_t, :][None]

        behavior_z = my_z_score(behavior)
        behavior_z = behavior_z[behav_vars][:,filter_t,:]
        global_activity_all = nanmean(activity, axis=0)[None][:,filter_t,:]
        global_activity_except = nanmean(delete(activity, (i_source, i_target), axis=0), axis=0)[None][:,filter_t,:]

        #Regress behaviour, global avg activity, avg activity except source and target or combinations of those
        Regressors = behavior_z
        # Regressors = global_activity_all
        # Regressors = global_activity_except
        # Regressors = vstack((behavior_z, global_activity_all))
        # Regressors = vstack((behavior_z, global_activity_except))

        Act_source = remove_confounder_separate_cues(Act_source, Regressors, cues)
        Act_target = remove_confounder_separate_cues(Act_target, Regressors, cues)

        activity_pair = vstack((Act_source, Act_target))
        noisecorr_EI[i] = compute_noise_corr(0, 1, activity_pair, corrects, cues, threshold, type_corr, how)
    else:
        Act_source = activity[i_source, filter_t,:][None]
        Act_target = activity[i_target, filter_t,:][None]
        activity_pair = vstack((Act_source, Act_target))
        noisecorr_EI[i] = compute_noise_corr(0, 1, activity_pair, corrects, cues, threshold, type_corr, how)

for i in range(n_IE):
    print(f'i_IE={i}')
    i_source = source_IE[i]
    i_target = target_IE[i]

    if filter_time:
        filter_t = get_good_times(i_source, i_target, activity)
        if filter_time == 'Max':
            if isnan(max_times_IE[i]):
                continue
            else:
                filter_t = [int(max_times_IE[i])]
    else:
        filter_t = range(activity.shape[1])

    if remove_confounder:

        Act_source = activity[i_source, filter_t, :][None]
        Act_target = activity[i_target, filter_t, :][None]

        behavior_z = my_z_score(behavior)
        behavior_z = behavior_z[behav_vars][:,filter_t,:]
        global_activity_all = nanmean(activity,axis=0)[None][:,filter_t,:]
        global_activity_except = nanmean(delete(activity, (i_source,i_target), axis=0), axis=0)[None][:,filter_t,:]

        #Regress behaviour, global avg activity, avg activity except source and target or combinations of those
        Regressors = behavior_z
        # Regressors = global_activity_all
        # Regressors = global_activity_except
        # Regressors = vstack((behavior_z, global_activity_all))
        # Regressors = vstack((behavior_z, global_activity_except))

        Act_source = remove_confounder_separate_cues(Act_source, Regressors, cues)
        Act_target = remove_confounder_separate_cues(Act_target, Regressors, cues)

        activity_pair = vstack((Act_source, Act_target))
        noisecorr_IE[i] = compute_noise_corr(0, 1, activity_pair, corrects, cues, threshold, type_corr, how)
    else:
        Act_source = activity[i_source, filter_t,:][None]
        Act_target = activity[i_target, filter_t,:][None]
        activity_pair = vstack((Act_source, Act_target))
        noisecorr_IE[i] = compute_noise_corr(0, 1, activity_pair, corrects, cues, threshold, type_corr, how)


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
ax.set_title(f'E-to-I {type_corr}-{how}{which_var},\n c={c:.2f}, p={p:.3f}', fontsize=8)

ax = fig.add_subplot(1,3,2)
x,y = noisecorr_EI, synden_EI
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'I-to-E {type_corr}-{how}{which_var},\n c={c:.2f}, p={p:.3f}', fontsize=8)

ax = fig.add_subplot(1,3,3)
x = concatenate((noisecorr_IE-nanmean(noisecorr_IE), noisecorr_EI-nanmean(noisecorr_EI)))
y = concatenate((synden_IE-nanmean(synden_IE), -synden_EI+nanmean(synden_EI)))
filter = isfinite(x) & isfinite(y)
x,y = x[filter], y[filter]
model = LinearRegression().fit(x[:,None], y)
scatter(x,y,s=10,c='k')
plot(x, x*model.coef_+model.intercept_, lw=1, c='k')
c,p = pearsonr(x,y)
ax.set_title(f'Pooled {type_corr}-{how}\n{which_var},\n c={c:.2f}, p={p:.3f}', fontsize=8)

tight_layout()

##

data = {'noisecorr_IE':noisecorr_IE,
        'noisecorr_EI':noisecorr_EI,
        'synden_IE':synden_IE,
        'synden_EI': synden_EI}

save(f'./data/noisecorr_{type_corr}_{how}_{rem}_{which_var}_filtt.npy', data)

##


##

