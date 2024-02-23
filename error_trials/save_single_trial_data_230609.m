%% ATK 220712
% Compare activity in error and correct trials
% Compare inhib to excitatory neurons

%% Load neuron dataframe 
workingDir = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/";
MN_path = fullfile(workingDir,"MN_DF_PPC.csv");
MN_df = readtable(MN_path);
skel_ids = MN_df.skeleton_id;
neuron_ids = skel_ids;
sessions_ROI = MN_df.sessions_ROI;
neuron_type = MN_df.type;
for n_ix = 1:length(neuron_type)
    I_ids(n_ix) = cell2mat(neuron_type(n_ix))=="non pyramidal";
    E_ids(n_ix) = cell2mat(neuron_type(n_ix))=="pyramidal";
end
max_idx = floor(MN_df.choiceMI_max_idx);
select_idx = MN_df.select_idx_MI;
pref_dir = MN_df.select_idx_MI>0;
%% Loop through sessions and load neuron activity
masterPath = "/Volumes/Aaron's PPC/ppc/2P_data/";
mouse = 'LD187';
mySessions = {'LD187_141216','LD187_141215','LD187_141214','LD187_141213'};
%'LD187_141212','LD187_141211','LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206'};
%syncedDataDir = fullfile(masterPath, 'code_workspace',mouse,'syncedData');   
syncedDataDir = '/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data/trialAlignedData/';

neuronAct = cell(length(skel_ids), length(mySessions));
neuronActNorm = cell(length(skel_ids), length(mySessions));
neuronActBin = cell(length(skel_ids), length(mySessions));
allCue = cell(1,length(mySessions));
allCueDir = cell(1,length(mySessions));
allChoice = cell(1,length(mySessions));
allCorrect = cell(1,length(mySessions));
for s_ix = 1:length(mySessions)
    session = cell2mat(mySessions(s_ix));
    disp(session)
    trialAlignedData = open(fullfile(syncedDataDir,[session '.mat'])).trialAlignedData;
    allCue{1,s_ix} = trialAlignedData.cueType;
    allCueDir{1,s_ix} = trialAlignedData.cueDir;
    allChoice{1,s_ix} = trialAlignedData.choice;
    allCorrect{1,s_ix} = trialAlignedData.isCorrect;
    for n_ix = 1:length(skel_ids)
        rois = parse_sessions_ROI(sessions_ROI{n_ix});
        if rois(s_ix) ~= -1
            mat_roi = rois(s_ix)+1; % MATLAB indexing from 1!
            neuronAct{n_ix, s_ix} = squeeze(trialAlignedData.CaData(mat_roi,:,:))';
            neuronAct_mean = nanmean(trialAlignedData.CaData(mat_roi,:,:),3);
            neuronAct_scale = nanmean(trialAlignedData.CaData(mat_roi,:,:),[2,3]);
            neuronActNorm{n_ix, s_ix} = neuronAct_mean/neuronAct_scale;
            neuronActBin{n_ix, s_ix} = squeeze(trialAlignedData.CaData(mat_roi,:,:) > 0)';
        end
    end
end

%% Concatenate activity for each neuron
neuronActCombined = cell(1,length(skel_ids));
neuronActNormCombined = cell(1,length(skel_ids));
neuronActBinCombined = cell(1,length(skel_ids));
cueCombined = cell(1,length(skel_ids));
cueDirCombined = cell(1,length(skel_ids));
choiceCombined = cell(1,length(skel_ids));
correctCombined = cell(1, length(skel_ids));
for n_ix = 1:length(skel_ids)
    for s_ix = 1:length(mySessions) % SELECT TRIALS HERE
        if ~isempty(neuronAct{n_ix,s_ix})
            neuronActCombined{n_ix} = cat(2, neuronActCombined{n_ix}, neuronAct{n_ix,s_ix});
            neuronActNormCombined{n_ix} = cat(2, neuronActNormCombined{n_ix}, neuronActNorm{n_ix,s_ix});
            neuronActBinCombined{n_ix} = cat(2, neuronActBinCombined{n_ix}, neuronActBin{n_ix,s_ix});
            cueCombined{n_ix} = cat(2, cueCombined{n_ix}, allCue{1,s_ix});
            cueDirCombined{n_ix} = cat(2, cueDirCombined{n_ix}, allCueDir{1,s_ix});
            choiceCombined{n_ix} = cat(2, choiceCombined{n_ix}, allChoice{1,s_ix});
            correctCombined{n_ix} = logical(cat(2, correctCombined{n_ix}, allCorrect{1,s_ix}));
        end
    end
end

%% Plot some tests

cell_idx = 5;
sess_idx = 2;
figure; plot(nanmean(neuronAct{cell_idx,sess_idx},2));

figure; plot(nanmean(neuronActCombined{1,cell_idx},2));
disp(num2str(neuron_ids(cell_idx)));
%% Save objects
workingDir = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/";
cn_path = fullfile(workingDir,"dir_cn_DF.csv");
connections_table = readtable(cn_path);

%% Save data for Stefano
save(fullfile(workingDir,'230717_single_trial_data'),'connections_table','neuronAct', ...
    'neuronActCombined','neuron_ids','E_ids','I_ids','allCue','allCueDir','cueCombined',...
    'cueDirCombined','allCorrect','correctCombined','-v7.3');

%%
function rois = parse_sessions_ROI(sess_str)
    my_str = strip(strip(sess_str, '['),']');
    my_cells = split(my_str," ");
    my_cells = my_cells(strlength(my_cells)>0);
    rois = cellfun(@str2num, my_cells);
end

function centile = prctileofscore(data,value)
    
    data = data(:)';
    value = value(:);
    
    nless = sum(data < value, 2);
    nequal = sum(data == value, 2);
    centile = 100 * (nless + 0.5.*nequal) / length(data);
end
    