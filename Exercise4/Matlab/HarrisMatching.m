%% Let's try to do Harris corner extraction and matching using our own
% implementation in a less black-box manner.
I1 = (imread('Boston1.png'));
I2 = (imread('Boston2m.png'));

% Harris corner extraction, take a look at the source code of harris.m
[x1,y1]=harris(I1);
[x2,y2]=harris(I2);

n=size(x1,1);
m=size(x2,1);

% Pre-allocate the memory for the 15*15 image patches extracted
% around each corner point from both images
w=7;
patchA=zeros(2*w+1,2*w+1,n);
patchB=zeros(2*w+1,2*w+1,m);

% The following part extracts the patches using bilinear interpolation
[X,Y]=meshgrid(1:size(I1,2),1:size(I1,1));
[Xp,Yp]=meshgrid(-w:w,-w:w);
mA=zeros(1,n);sA=zeros(1,n);
mB=zeros(1,m);sB=zeros(1,m);
for i=1:n
    patchA(:,:,i)=interp2(X,Y,double(I1),Xp+x1(i),Yp+y1(i),'*linear',0);
    mA(i)=sum(sum(patchA(:,:,i)))/((2*w+1)^2);
    sA(i)=sqrt(sum(sum((patchA(:,:,i)-mA(i)).^2)));
end
for j=1:m
    patchB(:,:,j)=interp2(X,Y,double(I2),Xp+x2(j),Yp+y2(j),'*linear',0);
    mB(j)=sum(sum(patchB(:,:,j)))/((2*w+1)^2);
    sB(j)=sqrt(sum(sum((patchB(:,:,j)-mB(j)).^2)));
end


%% SSD
% Compute the sum of squared differences (SSD) of pixels' intensities
% for all pairs of patches from the two images
SumOfSquaredDiff=zeros(n,m);
for i=1:n
    for j=1:m
        SumOfSquaredDiff(i,j)=sum(sum((patchA(:,:,i)-patchB(:,:,j)).^2));
    end
end

% Next, compute pairs of patches that are mutually nearest neighbors
% according to the measure
[ss2,ids2]=min(SumOfSquaredDiff,[],2);
[ss1,ids1]=min(SumOfSquaredDiff,[],1);
pairs=[];
for k=1:n
    if k==ids1(ids2(k))
        pairs=[pairs;k ids2(k) ss2(k)];
    end
end

% We sort the mutually nearest neighbors based on the score
[sorted_ssd,id_ssd]=sort(pairs(:,3),1,'ascend');

% Visualize the 40 best matches which are mutual nearest neighbors
Nvis=40;
montage=[I1 I2];
figure;imagesc(montage);axis image; colormap('gray');hold on
title('The best 40 matches according to SSD measure');
for k=1:min(length(id_ssd),Nvis)
    l=id_ssd(k);
    plot(x1(pairs(l,1)),y1(pairs(l,1)),'mx');
    plot(x2(pairs(l,2))+size(I1,2),y2(pairs(l,2)),'mx');
    plot([x1(pairs(l,1)); x2(pairs(l,2))+size(I1,2)],[y1(pairs(l,1)); y2(pairs(l,2))],'c-','LineWidth',1);
end


%% NCC
% Now, your task is to do matching in similar manner by using normalized
% cross-correlation (NCC) instead of SSD.
%
% HINT: Compared to the previous SSD-based implementation, all you need to
% do is to modify the lines performing SumOfSquaredDiff calculation.
% Thereafter, you can proceed as above but notice the following details:
% You need to determine the mutually nearest neighbors by finding pairs for
% which NCC maximized (i.e not minimized like SSD). Also, you need to sort
% the matches in descending order in terms of NCC in order to find the best
% matches (i.e not ascending order as with SSD).

%%-your-code-starts-here-%%

%%-your-code-ends-here-%%


% Next we visualize the 40 best matches which are mutual nearest neighbors
Nvis=40;
montage=[I1 I2];
figure;imagesc(montage);axis image; colormap('gray');hold on
title('The best 40 matches according to NCC measure');
for k=1:min(length(id_ncc), Nvis)
    l=id_ncc(k);
    plot(x1(pairs(l,1)),y1(pairs(l,1)),'mx');
    plot(x2(pairs(l,2))+size(I1,2),y2(pairs(l,2)),'mx');
    plot([x1(pairs(l,1)); x2(pairs(l,2))+size(I1,2)],[y1(pairs(l,1)); y2(pairs(l,2))],'c-','LineWidth',1);
end
