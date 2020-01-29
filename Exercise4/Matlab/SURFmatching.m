%% Load images and pre-computed descriptors
I1 = (imread('boat1.png'));
I2 = (imread('boat6.png'));

% Pre-computed descriptors, can be done with OpenCV or Matlab's built-in functions
load f1.mat
load f2.mat
load vpts1.mat
load vpts2.mat


%% Compute the pairwise distances of feature vectors to matrix 'distmat'
distmat=zeros(size(f1,1),size(f2,1));
for i=1:size(f1,1)
    for j=1:size(f2,1)
        distmat(i,j)=norm(f1(i,:)-f2(j,:));
    end
end

% Determine the mutually nearest neighbors
[dist2,ids2]=min(distmat,[],2);
[dist1,ids1]=min(distmat,[],1);
pairs=[];
for k=1:length(ids2)
    if k==ids1(ids2(k))
        pairs=[pairs;k ids2(k) dist2(k)];
    end
end

%% Nearest neighbor distance based ordering

% Sort the mutually nearest neighbors based on the distance 
nnd=pairs(:,3);  % distances for each pair


[snnd,id_nnd]=sort(nnd,1,'ascend');

% Visualize the 5 best matches 
Nvis=5;

fig1=figure;
imshow([I1  I2]);hold on;
title('The top 5 mutual nearest neighbors of SURF features');
    
t=[0:1:360]/180*pi;
idlist=id_nnd;
for k=1:Nvis
    pid1=pairs(idlist(k),1);
    pid2=pairs(idlist(k),2);
    
    loc1=vpts1_loc(pid1,:);
    r1=6*vpts1_scale(pid1);
    loc2=vpts2_loc(pid2,:);
    r2=6*vpts2_scale(pid2);
    
    figure(fig1);
    plot(loc1(1)+r1*cos(t),loc1(2)+r1*sin(t),'m-','LineWidth',3);
    plot(loc2(1)+r2*cos(t)+size(I1,2),loc2(2)+r2*sin(t),'m-','LineWidth',3);
    plot([loc1(1);loc2(1)+size(I1,2)],[loc1(2); loc2(2)],'c-');
end
% How many of the top 5 matches appear to be correct correspondences?

%% Nearest neighbor distance ratio based ordering
% Now, your task is to compute and visualize the top 5 matches based on 
% the nearest neighbor distance ratio defined in Equation (4.18) in the course book.
% How many of those are correct correspondences?
%
% HINT: Loop through the first column in 'pairs' (first feature vector
% indices), use each index value from this column to get the corresponding
% row from distmat_sorted, get the nearest and second nearest distances from 
% this row in order to calculate the distance ratio and store this to nndr.

distmat_sorted = sort(distmat, 2, 'ascend');  % each row sorted in ascending order
nndr=zeros(size(pairs,1),1);  % pre-allocate memory

%%-your-code-starts-here-%%)

%%-your-code-ends-here-%%

% Visualize the 5 best matches 
Nvis=5;

fig1=figure;
imshow([I1  I2]);hold on;
title('The top 5 mutual nearest neighbors of SURF features according to NNDR');
    
t=[0:1:360]/180*pi;
idlist=id_nndr;
for k=1:Nvis
    pid1=pairs(idlist(k),1);
    pid2=pairs(idlist(k),2);
    
    loc1=vpts1_loc(pid1,:);
    r1=6*vpts1_scale(pid1);
    loc2=vpts2_loc(pid2,:);
    r2=6*vpts2_scale(pid2);
    
    figure(fig1);
    plot(loc1(1)+r1*cos(t),loc1(2)+r1*sin(t),'m-','LineWidth',3);
    plot(loc2(1)+r2*cos(t)+size(I1,2),loc2(2)+r2*sin(t),'m-','LineWidth',3);
    plot([loc1(1);loc2(1)+size(I1,2)],[loc1(2); loc2(2)],'c-');
end

% How many of the top 5 matches appear to be correct correspondences?
