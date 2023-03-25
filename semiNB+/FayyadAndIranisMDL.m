function accepted  = FayyadAndIranisMDL(priorCounts, bestCounts, num_Inst, num_CutPoint,N_0)

prior_entropy = entropy(priorCounts);
e = entropyConditionedOnRows(bestCounts);

gain = prior_entropy - e;

num_classes = numel(find(priorCounts>0));

num_classLeft = numel(find(bestCounts(1,:)>0));
num_classRight = numel(find(bestCounts(2,:)>0));

entropyLeft = entropy(bestCounts(1,:));
entropyRight = entropy(bestCounts(2,:));

lambda = 1/(1+exp(-num_Inst/N_0));

delta = log2(power(3,num_classes) -2) - ...
    ((num_classes *prior_entropy) - (num_classRight*entropyRight) - (num_classLeft* entropyLeft));

accepted = (gain > lambda*(log2(num_CutPoint)+delta)/num_Inst);