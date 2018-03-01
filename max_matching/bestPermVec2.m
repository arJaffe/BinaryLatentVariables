function [cV, cnorm,perm,MatchingPermMat] = bestPermVec2(trueV, estimatedV)
% trueV and estimatedV are column vectors;
Q = size(trueV,1);
weights = zeros(Q, Q);
%trueV = reshape(trueA,[Q^2,1]);
%estimatedV = reshape(estimatedA,[Q^2,1]);
trueVmat = repmat(trueV,[1 Q]);
estimatedVmat = repmat(estimatedV,[1 Q]);
weights = (trueVmat - estimatedVmat').^2;
[Matching, Cost] = hungarian(weights);
cnorm = sqrt(Cost);
MatchingPermMat = zeros(Q,Q);
MatchingPermMat(sub2ind( [Q Q],   [1:Q], Matching)) = 1;
cV = MatchingPermMat * estimatedV;
% for i=1:Q
%     perm(i) = find(Matching(i,:));
% end
perm = Matching;