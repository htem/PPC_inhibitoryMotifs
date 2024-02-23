from all_imports import *

def my_pearson(x1,x2):
    if len==0:
        # print(x1,x2)
        return nan,1
    else:
        filter = isfinite(x1) & isfinite(x2)
        if sum(filter)<=2:
            return nan,1
        else:
            return pearsonr(x1[filter], x2[filter])

def my_pearson_2D(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    T = X1.shape[0]
    P = empty(T)
    for i in range(T):
        P[i] = my_pearson(X1[i], X2[i])[0]
    return P

def my_z_score(B):
    B_zscored = empty_like(B)
    for i in range(B.shape[0]):
        m = nanmean(B[i], (0,1))
        s = nanstd(B[i], (0,1))
        B_zscored[i] = (B[i] - m)/s
    return B_zscored

def weigthed_time_average(X):
    ncells, ntime, ntrials = X.shape
    Xavg = nanmean(X,axis=2)
    Z = empty((ncells, ntrials))
    for i_cell in range(ncells):
        W = Xavg[i_cell]
        W = W/nansum(W)
        Z[i_cell] = W @ X[i_cell]
    return Z

def remove_confounder_1D(X, Conf): # X is ncells x ntrials
    X_conf = empty_like(X)
    ncells = X.shape[0]
    Z = X.T
    Conf = Conf.T
    R2 = []
    for i in range(ncells):
        y = Z[:, i]
        filter = isfinite(Conf).all(axis=1) & isfinite(y)
        if y[filter].shape == (0,):
            X_conf[i] = y
        else:
            model = LinearRegression().fit(Conf[filter], y[filter])
            beta = model.coef_
            beta0 = model.intercept_
            X_conf[i] = y - Conf @ beta - beta0
    return X_conf

def remove_confounder_2D(X, Conf):
    assert X.shape[1] == Conf.shape[1]
    assert X.shape[2] == Conf.shape[2]
    X_conf = empty_like(X)
    ntime = X.shape[1]
    for i in range(ntime):
        X_conf[:,i,:] = remove_confounder_1D(X[:,i,:], Conf[:,i,:])
    return X_conf

def remove_confounder_separate_cues(activity, behavior, cues):
    assert cues.shape[0] == activity.shape[2]
    activity_conf = empty_like(activity)
    A1 = activity[:, :, cues == 2]
    A2 = activity[:, :, cues == 3]
    behavior1 = behavior[:, :, cues == 2]
    behavior2 = behavior[:, :, cues == 3]
    activity_conf[:, :, cues == 2] = remove_confounder_2D(A1, behavior1)
    activity_conf[:, :, cues == 3] = remove_confounder_2D(A2, behavior2)
    return activity_conf


def get_common_sessions(i_source, i_target, there_is_session):
    X = there_is_session[i_source] & there_is_session[i_target]
    return where(X==True)[0].tolist()

def get_data_for_noisecorr(X, corrects, cues, threshold, type_corr):
    assert cues.shape[0] == X.shape[2]
    ncells = X.shape[0]
    X_cue1 = X[:,:,cues==2]
    X_cue2 = X[:,:,cues==3]

    if type_corr == 'Avg':
        X_cue1 = nanmean(X_cue1, axis=1) # avg over time
        X_cue2 = nanmean(X_cue2, axis=1)
        R_cue1 = X_cue1 - nanmean(X_cue1, axis=1, keepdims=1) #subtract signal (avg over trials) --> ncells x ntrials
        R_cue2 = X_cue2 - nanmean(X_cue2, axis=1, keepdims=1)

    if type_corr == 'Time_trials_pooled':
        X_cue1 = X_cue1 - nanmean(X_cue1, axis=2, keepdims=1) # first subtract signal at each time point (avg over trials)
        X_cue2 = X_cue2 - nanmean(X_cue2, axis=2, keepdims=1)
        R_cue1 = X_cue1.reshape(X_cue1.shape[0], -1) # pool timepoints and trials together
        R_cue2 = X_cue2.reshape(X_cue2.shape[0], -1)

    if type_corr == 'Avg_inverted':
        X_cue1 = X_cue1 - nanmean(X_cue1, axis=2, keepdims=1) # first subtract signal at each time point (avg over trials)
        X_cue2 = X_cue2 - nanmean(X_cue2, axis=2, keepdims=1)
        R_cue1 = nanmean(X_cue1, axis=1) # average residuals over time
        R_cue2 = nanmean(X_cue2, axis=1)

    if type_corr == 'Avg_after' or type_corr == 'Max_after':
        R_cue1 = X_cue1 - nanmean(X_cue1, axis=2, keepdims=1) #subtract signal (avg over trials) --> ncells x ntime x ntrials
        R_cue2 = X_cue2 - nanmean(X_cue2, axis=2, keepdims=1)

    if type_corr == 'W_Avg':
        X_cue1 = weigthed_time_average(X_cue1) # --> ncells x ntrials
        X_cue2 = weigthed_time_average(X_cue2)
        R_cue1 = X_cue1 - nanmean(X_cue1, axis=1, keepdims=1)
        R_cue2 = X_cue2 - nanmean(X_cue2, axis=1, keepdims=1)

    if type_corr == 'Max':
        x1 = nanmean(X_cue1, axis=2)
        x2 = nanmean(X_cue2, axis=2)
        if x1.shape==(0,) or x2.shape==(0,):
            R_cue1=array([])
            R_cue2=array([])
        else:
            idx_cue1 = argmax( x1, axis=1 )
            idx_cue2 = argmax( x2, axis=1 )
            X_cue1 = X_cue1[range(ncells), idx_cue1]
            X_cue2 = X_cue2[range(ncells), idx_cue2]
            R_cue1 = X_cue1 - nanmean(X_cue1, axis=1, keepdims=1)
            R_cue2 = X_cue2 - nanmean(X_cue2, axis=1, keepdims=1)

    return R_cue1, R_cue2

def compute_noise_corr(i_source, i_target, X, corrects, cues, threshold, type_corr, how):

    R_cue1, R_cue2 = get_data_for_noisecorr(X, corrects, cues, threshold, type_corr)

    R1_cue1 = R_cue1[i_source]
    R2_cue1 = R_cue1[i_target]
    R1_cue2 = R_cue2[i_source]
    R2_cue2 = R_cue2[i_target]

    if type_corr == 'Avg_after':
        if how == 'Combined':
            R1 = concatenate((R1_cue1, R1_cue2), axis=1)
            R2 = concatenate((R2_cue1, R2_cue2), axis=1)
            P = my_pearson_2D(R1, R2)
            return nanmean(P)
        if how == 'Separate':
            P_cue1 = my_pearson_2D(R1_cue1, R2_cue1)
            P_cue2 = my_pearson_2D(R1_cue2, R2_cue2)
            return (nanmean(P_cue1)+nanmean(P_cue2))/2

    elif type_corr == 'Max_after':
        if how == 'Combined':
            R1 = concatenate((R1_cue1, R1_cue2), axis=1)
            R2 = concatenate((R2_cue1, R2_cue2), axis=1)
            P = my_pearson_2D(R1, R2)
            try:
                res = P[nanargmax(abs(P))]
            except:
                return nan
            else:
                return res
        if how == 'Separate':
            P_cue1 = my_pearson_2D(R1_cue1, R2_cue1)
            P_cue2 = my_pearson_2D(R1_cue2, R2_cue2)
            try:
                res1 = P_cue1[nanargmax(abs(P_cue1))]
                res2 = P_cue2[nanargmax(abs(P_cue2))]
            except:
                return nan
            else:
                return (res1+res2)/2

    else:
        if how == 'Combined':
            R1 = concatenate((R1_cue1, R1_cue2))
            R2 = concatenate((R2_cue1, R2_cue2))
            P = my_pearson(R1, R2)[0]
            return P
        if how == 'Separate':
            P_cue1 = my_pearson(R1_cue1, R2_cue1)[0]
            P_cue2 = my_pearson(R1_cue2, R2_cue2)[0]
            return (P_cue1+P_cue2)/2












