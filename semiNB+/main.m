clear;

%% load data
DATA = load ('../datasets/eswa_dataset_org/Cardio_2126_23_N.dat');
rng('default')
feat = DATA(:,1:end-1);
label = DATA(:,end);
class_lab = unique(label);

%% get 10-fold splitting
indices = crossvalind('Kfold',label,fold,'Classes',class_lab);
rates =zeros(fold,1);


for k=1:10
    %% get discretized data
    [tr_feat, tr_label, te_feat, te_label] = SADD(feat,label,indices,k);

    %% classification using NB classifier, get classification accuracy of testing set
    rates(k) = NB(tr_feat,tr_label,te_feat,te_label);

end
%% get average classification of 10-fold cv
avg = mean(rate);