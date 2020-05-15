clear all
close all

nSamples = 2000;
nVars = 2;
nLabeled = 100;
nNeighbors = 5;
maxIter = 30;
doRandom = 0;
doCyclic = 0;

% Generate Data
[x,yTrue] = GD_GenerateData(2, nSamples, nVars);
x = x';

% % Show true labels
% figure
% plot(x(yTrue==1,1),x(yTrue==1,2),'r.');hold on;
% plot(x(yTrue==2,1),x(yTrue==2,2),'b.');
% title('Ground Truth');

% Only label some of the points
perm = randperm(nSamples);
labeled = perm(1:nLabeled);
y = zeros(nSamples,1);
y(labeled) = yTrue(labeled);

% % Show data
% figure;
% plot(x(:,1),x(:,2),'k.');hold on;
% h=plot(x(y==1,1),x(y==1,2),'ro','MarkerFaceColor','r');
% plot(x(y==2,1),x(y==2,2),'bo','MarkerFaceColor','b');
% title('Training Data');

% Compute all (squared) distances
D = x.^2*ones(nVars,nSamples) + ones(nSamples,nVars)*(x').^2 - 2*x*x';

% Connect only if in KNN graph of one or the other
[sorted,sortedInd] = sort(D);
W = zeros(nSamples);
for s = 1:nSamples
    W(sortedInd(2:nNeighbors+1,s),s) = 1;
end
W = W+W';
W(W>1) = 1;

save('W_y.mat', 'W', 'y', '-mat7-binary')
