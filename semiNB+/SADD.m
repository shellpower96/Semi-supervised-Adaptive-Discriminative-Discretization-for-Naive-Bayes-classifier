function [tr_feat,tr_label,te_feat,te_label] = SADD(feat,label,indices,k)
%% parameter setting
N_0 = 2000;
knn_K =1;
%% get the training set and test set
test_id = find(indices ==k);
train_id = find(indices ~=k);

te_feat = feat(test_id,:);
te_label = label(test_id,:);

tr_feat = feat(train_id,:);
tr_label = label(train_id,:);

%% get the unlabeled set
U_feat = te_feat;
%% pseudo-labeling using k-NN
disp('Pseudo labeling')
reverseStr = '';
NUM_U_SAMPLE = size(U_feat,1);
pseudo_label = zeros(NUM_U_SAMPLE,1);
for i=1:NUM_U_SAMPLE
    id = knnsearch(tr_feat,U_feat(i,:),'K',knn_K);
    temp_label = tr_label(id);
    [ia,~,ic] = unique(temp_label);
    mWeight = accumarray(ic,1);
    [~,id_x] = max(mWeight);
    pseudo_label(i) = ia(id_x);
    %show process
    percentDone = 100 * i / NUM_U_SAMPLE;
    msg = sprintf('Percent done: %3.1f', percentDone);
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end
%% get the full training set after pseudo-labeling
semi_feat = [tr_feat;te_feat];
semi_label = [tr_label;pseudo_label];
%% derive the discretization scheme vis adaptive discriminative discretization for each feature
disp('Discretizing')
[rows,cols] = size(semi_feat);
m_cutPoints = zeros(cols,rows);
count = zeros(cols,1);
reverseStr = '';
for j = 1:cols
    attribute = semi_feat(:,j);
    [A,I] = sort(attribute);
    labels = semi_label(I);
    temp = cutPointsForSubset(A,labels,1,rows+1,N_0);
    count(j) = numel(temp);
    m_cutPoints(j,1:count(j))= temp;
    %show process
    percentDone = 100 * j / cols;
    msg = sprintf('Percent done: %3.1f', percentDone);
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end

%% get discretzied data of training data
[rows,cols] = size(tr_feat);
new_tr_feat = zeros(rows,cols);
for i = 1:rows
    for j = 1:cols
        cutPoint = m_cutPoints(j,1:count(j));
        [~,idx] =min(abs(tr_feat(i,j)-cutPoint));
        if numel(cutPoint) ==0
            new_tr_feat(i,j) = 1;
        else
            if tr_feat(i,j) <= cutPoint(idx)
                new_tr_feat(i,j) = idx;
            else
                new_tr_feat(i,j) = idx+1;
            end
        end
    end
end
%% get discretzied data of test data
[rows,cols] =size(te_feat);
new_te_feat = zeros(size(te_feat));
for i = 1:rows
    for j = 1:cols
        cutPoint = m_cutPoints(j,1:count(j));
        [~,idx] =min(abs(te_feat(i,j)-cutPoint));
        if numel(cutPoint) ==0
            new_te_feat(i,j) = 1;
        else
            if te_feat(i,j) <= cutPoint(idx)
                new_te_feat(i,j) = idx;
            else
                new_te_feat(i,j) = idx+1;
            end
        end
    end
end

tr_feat = new_tr_feat;
te_feat = new_te_feat;

end