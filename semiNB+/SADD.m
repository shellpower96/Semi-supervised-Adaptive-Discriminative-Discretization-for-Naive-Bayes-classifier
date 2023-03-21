function [tr_feat,tr_label,te_feat,te_label] = SADD(feat,label,indices,k)
%% parameter setting
N_0 = 2000;
knn_K =1;

test_id = find(indices ==k);
train_id = find(indices ~=k);

te_feat = feat(test_id,:);
te_label = label(test_id,:);

tr_feat = feat(train_id,:);
tr_label = label(train_id,:);

U_feat = te_feat;

NUM_U_SAMPLE = size(U_feat,1);
pseudo_label = zeros(NUM_U_SAMPLE,1);
for i=1:NUM_U_SAMPLE
    id = knnsearch(tr_feat,U_feat(i,:),'K',knn_K);
    temp_label = tr_label(id);
    [ia,~,ic] = unique(temp_label);
    mWeight = accumarray(ic,1);
    [~,id_x] = max(mWeight);
    pseudo_label(i) = ia(id_x);
end

semi_feat = [tr_feat;te_feat];
semi_label = [tr_label;pseudo_label];

[rows,cols] = size(semi_feat);
m_cutPoints = zeros(cols,rows);
count = zeros(cols,1);
for j = 1:cols
    attribute = semi_feat(:,j);
    [A,I] = sort(attribute);
    labels = semi_label(I);
    temp = cutPointsForSubset(A,labels,1,rows+1,1,N_0);
    count(j) = numel(temp);
    m_cutPoints(j,1:count(j))= temp;
end

%% discretization on training data
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