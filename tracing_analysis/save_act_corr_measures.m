%% 2005012 

% Load activity data from disk. Save only activity correlation data
% and trial average data for ROIs.
output_file = '/n/groups/htem/temcagt/datasets/ppc/2P_data/code_workspace/LD187/summaryData/actCorr.mat';

% Run from ppc_project_analysis repository folder
sessions = {'LD187_141216','LD187_141215','LD187_141214',...
        'LD187_141213','LD187_141212','LD187_141211',...
        'LD187_141210','LD187_141209',...
        'LD187_141208','LD187_141207','LD187_141206'};
 
data = struct;
actCorr = struct;
preprocess_output_dir = fullfile(masterPath, 'code_workspace',mouse,'syncedData');   

for sidx = 1:length(sessions)
    session = sessions{sidx};
    disp(['Loading Session ' sessions{sidx}]);
    data.(session) = load(fullfile(preprocess_output_dir,session),'trialAlignedData');
    actCorr.(session).trialAlignedData.RL_selectIdx = data.(session).trialAlignedData.RL_selectIdx;
    actCorr.(session).trialAlignedData.SNR_raw = data.(session).trialAlignedData.SNR_raw;
    
    actCorr.(session).trialAlignedData.corr_all = data.(session).trialAlignedData.corr_all;
    actCorr.(session).trialAlignedData.corr_Ca_concat = data.(session).trialAlignedData.corr_Ca_concat;
    actCorr.(session).trialAlignedData.corr_Ca_trialMean = data.(session).trialAlignedData.corr_Ca_trialMean;
    actCorr.(session).trialAlignedData.corr_Ca_residual = data.(session).trialAlignedData.corr_Ca_residual;
        
    actCorr.(session).trialAlignedData.bR_trials = struct;
    actCorr.(session).trialAlignedData.bR_trials.trial_snr = data.(session).trialAlignedData.bR_trials.trial_snr;
    actCorr.(session).trialAlignedData.bR_trials.Ca_trialMean = data.(session).trialAlignedData.bR_trials.Ca_trialMean;
    actCorr.(session).trialAlignedData.bR_trials.corrcoef = data.(session).trialAlignedData.bR_trials.corrcoef;
    actCorr.(session).trialAlignedData.bR_trials.tCOM = data.(session).trialAlignedData.bR_trials.tCOM;
    
    actCorr.(session).trialAlignedData.wL_trials = struct;
    actCorr.(session).trialAlignedData.wL_trials.trial_snr = data.(session).trialAlignedData.wL_trials.trial_snr;
    actCorr.(session).trialAlignedData.wL_trials.Ca_trialMean = data.(session).trialAlignedData.wL_trials.Ca_trialMean;
    actCorr.(session).trialAlignedData.wL_trials.corrcoef = data.(session).trialAlignedData.wL_trials.corrcoef;
    actCorr.(session).trialAlignedData.wL_trials.tCOM = data.(session).trialAlignedData.wL_trials.tCOM;
end
save(output_file,'actCorr','-v7.3');
