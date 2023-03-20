function rate = NB(tr_feat, tr_label, te_feat, te_label)

    NUM_CLASSES = numel(class_lab);
    NUM_TE_ATTRI = numel(te_label);
    prior_prob =zeros(NUM_CLASSES,1);
    predt_label = zeros(NUM_TE_ATTRI,1);
    for c = 1:NUM_CLASSES
        prior_prob(c) = (numel(find(tr_label == c))+ 1/numel(class_lab))/(numel(tr_label)+1);
    end

    for i =1:NUM_TE_ATTRI
        prob = zeros(cols,NUM_CLASSES);
        for j = 1:cols
            attribute = tr_feat(:,j);
            for c =1:NUM_CLASSES
                [m_count,~,ic] = unique(attribute(tr_label==c));
                mWeight = accumarray(ic,1);
                id = find(m_count == te_feat(i,j));
                if isempty(mWeight)
                    prob(j,c) = 1/numel(unique(attribute))/(sum(mWeight)+1);
                else
                    if isempty(id)
                        prob(j,c) = 1/numel(unique(attribute))/(sum(mWeight)+1);
                    else
                        prob(j,c) = (mWeight(id)+1/numel(unique(attribute)))/ (sum(mWeight)+1);
                    end
                end
            end
        end
        post_prob = prior_prob'.*prod(prob);

        [~,id] = max(post_prob);
        predt_label(i) = id;
    end
    rate = numel(find(predt_label ==te_label))/NUM_TE_ATTRI;
end