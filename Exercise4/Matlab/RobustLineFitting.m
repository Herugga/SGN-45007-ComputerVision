%% Load and plot points
load('points.mat','x','y');
figure;hold on;
plot(x,y,'kx');
axis equal

%% RANSAC parameters
% m is the number of data points
m=length(x);
% s is the size of the random sample
s=2;
% t is the inlier distance threshold
t=sqrt(3.84)*2;
% e is the expected outlier ratio
e=0.8;
% at least one random sample should be free from outliers with probability p 
p=0.999;
% required number of samples
N_estimated=log(1-p)/log(1-(1-e)^s);

%% RANSAC loop
% First initialize some variables
N=inf;
sample_count=0;
max_inliers=0;
best_line=zeros(3,1);

while(N>sample_count)
    % Pick two random samples
    a=ceil(m*rand(1));  % sample id 1
    b=ceil(m*rand(1));  % sample id 2
    if a==b %if the same point is drawn twice, must draw again
        continue;
    end
    
    % Determine the line crossing the points with the cross product of the points (in homogeneous coordinates).
    % Also normalize the line by dividing each element by sqrt(a^2+b^2)
    
    %%-your-code-starts-here%%

    %%-your-code-ends-here%%
    
    % Determine the inliers by finding the indices for the line and data
    % point dot products (absolute value) which are less than inlier threshold.
    
    %%-your-code-starts-here%%
    %% Compute the distances between all points with the fitting line 
     kLine = sample(:,2)-sample(:,1);% two points relative distance
     kLineNorm = kLine/norm(kLine);
     normVector = [-kLineNorm(2),kLineNorm(1)];%Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
     distance = normVector*(data - repmat(sample(:,1),1,number));
 %% Compute the inliers with distances smaller than the threshold
     inlierIdx = find(abs(distance)<=threshDist);
     inlierNum = length(inlierIdx);
    %%-your-code-ends-here%%
    
    % keep the hypothesis giving most inliers so far
    inlier_count=length(inliers);
    if inlier_count>max_inliers
        best_line=l(:);
    end
    
    % update the estimate of the outlier ratio
    e=1-inlier_count/m;
    % update the estimate for the required number of samples
    N=log(1-p)/log(1-(1-e)^s);
    
    sample_count=sample_count+1;
end

% Least squares fitting to the inliers of the best hypothesis, i.e.
% find the inliers similarly as above but this time for the best line.

%%-your-code-starts-here%%
%% Plot the best fitting line
 xAxis = -number/2:number/2; 
 yAxis = bestParameter1*xAxis + bestParameter2;
 plot(xAxis,yAxis,'r-','LineWidth',2);
%%-your-code-ends-here%%

% Fit a line to the given points (non-homogeneous)
l=linefitlsq(x_inliers, y_inliers);

% plot the resulting line and the inliers
k=-l(1)/l(2);
b=-l(3)/l(2);
plot(1:100,k*[1:100]+b,'m-');
plot(x(inliers),y(inliers),'ro');
