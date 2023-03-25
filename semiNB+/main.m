clear;

%% load data
name =''; % put the relative path for the dataset
DATA = load ('../IndoorLoc/indoorLoca_all.csv');
rng('default')
feat = DATA(:,1:end-1);
label = DATA(:,end);
class_lab = unique(label);

%% get 10-fold splitting
indices = crossvalind('Kfold',label,10,'Classes',class_lab);
rates =zeros(10,1);


for k=1:10
    %% get discretized data
    disp('SADD')
    [tr_feat, tr_label, te_feat, te_label] = SADD(feat,label,indices,k);
    disp('NB classification')
    %% classification using NB classifier, get classification accuracy of testing set
    rates(k) = NB(tr_feat,tr_label,te_feat,te_label);

end
%% get average classification of 10-fold cv
avg = mean(rate);