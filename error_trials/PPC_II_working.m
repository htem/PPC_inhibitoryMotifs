%% ATK 220708
% Working code for caculating intersection information
% Compare inhib to excitatory neurons
% Using intersection information from Marc Celotto (nit_ii)

%% Load neuron dataframe 
workingDir = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/";
MN_path = fullfile(workingDir,"MN_DF_PPC.csv");
MN_df = readtable(MN_path);
skel_ids = MN_df.skeleton_id;
sessions_ROI = MN_df.sessions_ROI;
neuron_type = MN_df.type;
max_idx = floor(MN_df.choiceMI_max_idx);
select_idx = MN_df.select_idx_MI;
pref_dir = MN_df.select_idx_MI>0;
%% Loop through sessions and load neuron activity
masterPath = "/Volumes/Aaron's PPC/ppc/2P_data/";
mouse = 'LD187';
mySessions = {'LD187_141216','LD187_141215','LD187_141214','LD187_141213','LD187_141212',...
    'LD187_141211','LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206'};
syncedDataDir = fullfile(masterPath, 'code_workspace',mouse,'syncedData');   

neuronAct = cell(length(skel_ids), length(mySessions));
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
            neuronActBin{n_ix, s_ix} = squeeze(trialAlignedData.CaData(mat_roi,:,:) > 0)';
        end
    end
end

%% Concatenate activity for each neuron
neuronActCombined = cell(1,length(skel_ids));
neuronActBinCombined = cell(1,length(skel_ids));
cueCombined = cell(1,length(skel_ids));
cueDirCombined = cell(1,length(skel_ids));
choiceCombined = cell(1,length(skel_ids));
correctCombined = cell(1, length(skel_ids));
for n_ix = 1:length(skel_ids)
    for s_ix = 1:length(mySessions)
        if ~isempty(neuronAct{n_ix,s_ix})
            neuronActCombined{n_ix} = cat(2, neuronActCombined{n_ix}, neuronAct{n_ix,s_ix});
            neuronActBinCombined{n_ix} = cat(2, neuronActBinCombined{n_ix}, neuronActBin{n_ix,s_ix});
            cueCombined{n_ix} = cat(2, cueCombined{n_ix}, allCue{1,s_ix});
            cueDirCombined{n_ix} = cat(2, cueDirCombined{n_ix}, allCueDir{1,s_ix});
            choiceCombined{n_ix} = cat(2, choiceCombined{n_ix}, allChoice{1,s_ix});
            correctCombined{n_ix} = logical(cat(2, correctCombined{n_ix}, allCorrect{1,s_ix}));
        end
    end
end

%% Calculate MI and II
opts = struct('n_binsX',2,'n_binsY',2,'bin_methodX','eqspace',...
    'bin_methodY','eqspace','method','dr','bias','naive');
% use maxMI timepoint for now
cueMI = nan(1,length(skel_ids));
cueMI_correct = nan(1,length(skel_ids));
cueMI_error = nan(1,length(skel_ids));
choiceMI_correct = nan(1,length(skel_ids));
choiceMI_error = nan(1,length(skel_ids));
choiceMI = nan(1,length(skel_ids));
neuronsII = nan(1,length(skel_ids));
pref_correct = nan(1,length(skel_ids));
pref_error = nan(1,length(skel_ids));
nonpref_correct = nan(1,length(skel_ids));
nonpref_error = nan(1,length(skel_ids));
pref_all = nan(1,length(skel_ids));
nonpref_all = nan(1,length(skel_ids));
E_pref = nan(1,length(skel_ids));
E_nonpref = nan(1,length(skel_ids));

for n_ix = 1:length(skel_ids)
    tpt = max_idx(n_ix); % MATLAB indexing already
    cue = cueCombined{n_ix};
    cueDir = cueDirCombined{n_ix};
    choice = choiceCombined{n_ix};
    og_trials = ismember(cue,[2,3]);
    isCorrect = correctCombined{n_ix};
    og_correct = isCorrect & og_trials;
    og_error = ~isCorrect & og_trials;
    
    og_pref_correct = isCorrect & og_trials & cueDir == pref_dir(n_ix);
    og_pref_error = ~isCorrect & og_trials & cueDir == pref_dir(n_ix);
    og_pref_all = og_trials & cueDir == pref_dir(n_ix);
    og_nonpref_correct = isCorrect & og_trials & cueDir ~= pref_dir(n_ix);
    og_nonpref_error = ~isCorrect & og_trials & cueDir ~= pref_dir(n_ix);     
    og_nonpref_all = og_trials & cueDir ~= pref_dir(n_ix);
    
    myAct = neuronActCombined;
    pref_correct(n_ix) = mean(myAct{n_ix}(tpt,og_pref_correct));
    pref_error(n_ix) = mean(myAct{n_ix}(tpt,og_pref_error));
    pref_all(n_ix) = mean(myAct{n_ix}(tpt,og_pref_all));
    nonpref_correct(n_ix) = mean(myAct{n_ix}(tpt,og_nonpref_correct));
    nonpref_error(n_ix) = mean(myAct{n_ix}(tpt,og_pref_error));
    nonpref_all(n_ix) =  mean(myAct{n_ix}(tpt,og_nonpref_all));
    
    E_pref(n_ix) = (pref_error(n_ix)-pref_correct(n_ix))/ pref_all(n_ix);
    E_nonpref(n_ix) = (nonpref_error(n_ix)-nonpref_correct(n_ix))/nonpref_all(n_ix);
    
    %use max timepoint and only orig cues (2,3) 
    X = squeeze(neuronActCombined{n_ix}(tpt,:));
    if ~any(isnan(X))
        cueMI(n_ix) = cell2mat(information(X(og_trials),cue(og_trials),opts,{'I'}));
        %choiceMI(n_ix) = cell2mat(information(X(og_trials),choice(og_trials),opts,{'I'}));
        %neuronsII(n_ix) = II(cue(og_trials), X(og_trials), choice(og_trials),opts);

        cueMI_correct(n_ix) = cell2mat(information(X(og_correct),cue(og_correct),opts,{'I'}));
        cueMI_error(n_ix) = cell2mat(information(X(og_error),cue(og_error),opts,{'I'}));

        %choiceMI_correct(n_ix) = cell2mat(information(X(og_correct),choice(og_correct),opts,{'I'}));
        %choiceMI_error(n_ix) = cell2mat(information(X(og_error),choice(og_error),opts,{'I'}));
    else
        disp(['NaNs found in ix :' num2str(n_ix)]);
    end
    

end

%% Make some plots?
figure; plot(abs(select_idx), cueMI, 'o');

figure; histogram(cueMI_correct-cueMI_error); 
title('cueMI correct - cueMI error');

%%
figure; histogram(cueMI -  neuronsII);
%% 
masterPath = "/Volumes/Aaron's PPC/ppc/2P_data/";
outputPath = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/2P_data"
mouse = 'LD187';
mySessions = {'LD187_141216','LD187_141215','LD187_141214','LD187_141213','LD187_141212',...
    'LD187_141211','LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206'};
%mySessions = {'LD187_141216'};
i = 1;
session = cell2mat(mySessions(i));
% Load trial aligned data
syncedDataDir = fullfile(masterPath, 'code_workspace',mouse,'syncedData');   
trialAlignedData = open(fullfile(syncedDataDir,[session '.mat'])).trialAlignedData;
%% Load Single Trial cue, choice, activity

% Only include original cues (types 2 and 3)
trialTypes = trialAlignedData.trialType;
trialTypes(1,1) = nan; %exclude first trial (has nans in ITI) 

virmenData = permute(trialAlignedData.virmenData(:,ismember(trialTypes,[2,3]),:),[1,3,2]);
cue = trialAlignedData.cueDir(ismember(trialTypes,[2,3]));
choice = trialAlignedData.choice(ismember(trialTypes,[2,3]));
isCorrect = trialAlignedData.isCorrect(ismember(trialTypes,[2,3]));
actData = permute(trialAlignedData.CaData(:,ismember(trialTypes,[2,3]),:),[1,3,2]);
actData = double(actData>0);


%% Test nit_ii mi calc
tpt = 27;
roi = 1;

X = squeeze(actData(roi, tpt,:))';

opts = struct('n_binsX',2,'n_binsY',2,'bin_methodX','eqspace',...
    'bin_methodY','eqspace','method','dr','bias','naive');
outputlist = {'I'};
information(X,cue,opts,{'I'})
information(X,choice,opts,{'I'})
II(cue, X, choice,opts)

%% Calc Mutual Information with trial type 
% requires https://github.com/nmtimme/Neuroscience-Information-Theory-Toolbox
trialTypes = trialAlignedData.trialType;
trialTypes(1,1) = nan; %exclude first trial (has nans in ITI) 
 % (A) Cue-based (wL and bR only, types 2,3)
disp('Calculating cue-based wL/bR')
DataRaster = cell(1);
% 1 - Ca Data
DataRaster{1} = permute(trialAlignedData.CaData(:,ismember(trialTypes,[2,3]),:),[1,3,2]);
% 2 - trial types
cueTypes = repmat(trialTypes(:,ismember(trialTypes, [2,3]))', [1,76])';
DataRaster{2}(1,:,:) = cueTypes==2;
disp([num2str(size(DataRaster{2},3)) ' trials included']);
cueMI = calc_MI(DataRaster);
%%

function rois = parse_sessions_ROI(sess_str)
    my_str = strip(strip(sess_str, '['),']');
    my_cells = split(my_str," ");
    my_cells = my_cells(strlength(my_cells)>0);
    rois = cellfun(@str2num, my_cells);
end
%%
function MI = calc_MI(DataRaster)
    % Binary binning (active or not)
    numBins=2;
    numROIs = length(DataRaster{1});
    StatesRaster{1} = int8(DataRaster{1}>0);
    StatesRaster{2} = DataRaster{2};

    % Calculate MI with trial type
    MI = nan(numROIs, 76);
    for id = 1:numROIs
        for t = 1:76
            VariableIDs = {1,id,t;2,1,1};
            MI(id, t) = instinfo(StatesRaster, 'PairMI', VariableIDs);
        end
    end
end



