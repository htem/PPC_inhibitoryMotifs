%% ATK 220712
% Compare activity in error and correct trials
% Compare inhib to excitatory neurons

%% Load neuron dataframe 
workingDir = "/Users/akuan/Dropbox (HMS)/htem_team/projects/PPC_project/analysis_dataframes/";
MN_path = fullfile(workingDir,"MN_DF_PPC.csv");
MN_df = readtable(MN_path);
skel_ids = MN_df.skeleton_id;
sessions_ROI = MN_df.sessions_ROI;
neuron_type = MN_df.type;
for n_ix = 1:length(neuron_type)
    I_idx(n_ix) = cell2mat(neuron_type(n_ix))=="non pyramidal";
    E_idx(n_ix) = cell2mat(neuron_type(n_ix))=="pyramidal";
end
max_idx = floor(MN_df.choiceMI_max_idx);
select_idx = MN_df.select_idx_MI;
pref_dir = MN_df.select_idx_MI>0;
%% Loop through sessions and load neuron activity
masterPath = "/Volumes/Aaron's PPC/ppc/2P_data/";
mouse = 'LD187';
mySessions = {'LD187_141216','LD187_141215','LD187_141214','LD187_141213','LD187_141212',...
    'LD187_141211','LD187_141210','LD187_141209','LD187_141208','LD187_141207','LD187_141206'};
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

%% Calculate metrics
opts = struct('n_binsX',2,'n_binsY',2,'bin_methodX','eqspace',...
    'bin_methodY','eqspace','method','dr','bias','naive');
n_shuf_max = 500;
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
E_pref_shuf = nan(n_ix, n_shuf_max);
E_nonpref_shuf = nan(n_ix, n_shuf_max);
E_pref_prc = nan(1,length(skel_ids));
E_nonpref_prc = nan(1, length(skel_ids));

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
    
    myAct = neuronActNormCombined;
    %{
    pref_correct(n_ix) = mean(myAct{n_ix}(tpt,og_pref_correct));
    pref_error(n_ix) = mean(myAct{n_ix}(tpt,og_pref_error));
    pref_all(n_ix) = mean(myAct{n_ix}(tpt,og_pref_all));
    nonpref_correct(n_ix) = mean(myAct{n_ix}(tpt,og_nonpref_correct));
    nonpref_error(n_ix) = mean(myAct{n_ix}(tpt,og_pref_error));
    nonpref_all(n_ix) =  mean(myAct{n_ix}(tpt,og_nonpref_all));
    %}
    if ~isempty(myAct{n_ix})
        if sum(og_pref_error) > 3 && sum(og_nonpref_error) > 3
            % Try averaging over whole trial
            pref_correct(n_ix) = mean(myAct{n_ix}(:,og_pref_correct),[1,2]);
            pref_error(n_ix) = mean(myAct{n_ix}(:,og_pref_error),[1,2]);
            pref_all(n_ix) = mean(myAct{n_ix}(:,og_pref_all),[1,2]);
            nonpref_correct(n_ix) = mean(myAct{n_ix}(:,og_nonpref_correct),[1,2]);
            nonpref_error(n_ix) = mean(myAct{n_ix}(:,og_nonpref_error),[1,2]);
            nonpref_all(n_ix) =  mean(myAct{n_ix}(:,og_nonpref_all),[1,2]);

            % Shuf trials for stats
            for n_shuf = 1:n_shuf_max
                myAct_shuf = myAct{n_ix}(1,randperm(length(myAct{n_ix})));
                pref_correct_shuf(n_ix, n_shuf) = mean(myAct_shuf(:,og_pref_correct),[1,2]);
                pref_error_shuf(n_ix, n_shuf) = mean(myAct_shuf(:,og_pref_error),[1,2]);
                pref_all_shuf(n_ix, n_shuf) = mean(myAct_shuf(:,og_pref_all),[1,2]);
                nonpref_correct_shuf(n_ix, n_shuf) = mean(myAct_shuf(:,og_nonpref_correct),[1,2]);
                nonpref_error_shuf(n_ix, n_shuf) = mean(myAct_shuf(:,og_nonpref_error),[1,2]);
                nonpref_all_shuf(n_ix, n_shuf) =  mean(myAct_shuf(:,og_nonpref_all),[1,2]);
                E_pref_shuf(n_ix, n_shuf) = (pref_error_shuf(n_ix, n_shuf)-pref_correct_shuf(n_ix, n_shuf))/ pref_all_shuf(n_ix, n_shuf);
                E_nonpref_shuf(n_ix, n_shuf) = (nonpref_error_shuf(n_ix, n_shuf)-nonpref_correct_shuf(n_ix, n_shuf))/ nonpref_all_shuf(n_ix, n_shuf);
            end

            E_pref(n_ix) = (pref_error(n_ix)-pref_correct(n_ix))/ pref_all(n_ix);
            E_nonpref(n_ix) = (nonpref_error(n_ix)-nonpref_correct(n_ix))/ nonpref_all(n_ix);
            E_pref_prc(n_ix) = prctileofscore(E_pref_shuf(n_ix,:), E_pref(n_ix));
            E_nonpref_prc(n_ix) = prctileofscore(E_nonpref_shuf(n_ix,:), E_nonpref(n_ix));
            
            E_pref_high(n_ix) = prctile(E_pref_shuf(n_ix, :),97.5);
            E_pref_low(n_ix) = prctile(E_pref_shuf(n_ix, :),2.5);
            E_nonpref_high(n_ix) = prctile(E_nonpref_shuf(n_ix, :),97.5);
            E_nonpref_low(n_ix) = prctile(E_nonpref_shuf(n_ix, :),2.5);
        end
    end
end

%% Make some plots?
figure;
g1 = repmat({'Exc Pref'},sum(E_idx),1);
g2 = repmat({'Inh Pref'},sum(I_idx),1);
g3 = repmat({'Exc NonPref'},sum(E_idx),1);
g4 = repmat({'Inh NonPref'},sum(I_idx),1);
g = [g1; g2; g3; g4];
x=[E_pref(E_idx)'; E_pref(I_idx)'; E_nonpref(E_idx)'; E_nonpref(I_idx)'];
figure; boxplot(x,g);
ylim([-1, 5]);

%% Count sig 

disp(num2str(sum(E_pref<E_pref_low)));
disp(num2str(sum(E_nonpref>E_nonpref_high)));
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
