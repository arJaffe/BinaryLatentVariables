function [cA, cnorm,perm] = bestPermMat(trueA, estimatedA)
Q = size(trueA,1);
permMat = eye(Q);
BestErr = Inf;
for i = 1 : Q
   for j = i : Q
       [cV, cnorm,perm, Matching] = bestPermVec2(trueA(:,i), estimatedA(:,j));
       TempErr = norm(Matching*estimatedA * Matching' - trueA);
       if TempErr < BestErr
           BestErr = TempErr;
           permMat = Matching;
       end
   end
end
cA = permMat * estimatedA * permMat';
 for i=1:Q
      perm(i) = find(permMat(i,:));
 end