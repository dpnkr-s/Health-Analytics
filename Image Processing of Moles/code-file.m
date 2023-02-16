%% Image processing of Moles
% In this lab, image of mole is processed first with k-means clustering
% algorithm to reduce the number of colors to 4 and then boundary of
% objects in new image is identified using ActiveContours matlab app.

%% Low-risk mole

A1 = imread('low_risk_1.jpg');
figure
subplot(1,2,1);
imshow(A1,'Border','tight');
title('Camera image');

% reshaping 3D matrix into 2D
[s11, s21, s31]=size(A1);
s1=s11*s21;
B1=double(reshape(A1,s1,s31));

% standardizing image data matrix
[N1,F1] = size(B1);
Bmean1 = mean(B1); Bvar1 = var(B1); o = ones(N1,1);
b1 = (B1-o*Bmean1)./sqrt(o*Bvar1);

% kmeans algorithm giving cluster number of each row index and the mean
% vectors of each cluster
k = 4; % number of clusters
[idx1, C1] = kmeans(b1,k);
oc = ones(k,1);
% obtaining true value of colors and replacing row belonging to a cluster
% with mean vector of a respective cluster
C1r = (C1.*sqrt(oc*Bvar1))+oc*Bmean1;
for i=1:k
    rows = find(idx1 == i);
    or = ones(length(rows),1);
    B1new(rows,:) = or.*C1r(i,:);
end

B1new=floor(B1new);
A1new=reshape(uint8(B1new),s11,s21,s31);
imwrite(A1new, 'low_risk_1new.jpg');

subplot(1,2,2);
imshow(A1new,'Border','tight');
title('After clustering with k=4')
suptitle('Low risk mole');
print('low_risk','-dpng')
%% Medium-risk mole

A2 = imread('medium_risk_11.jpg');
figure
subplot(1,2,1);
imshow(A2,'Border','tight');
title('Camera image');
[s12, s22, s32]=size(A2);
s2=s12*s22;
B2=double(reshape(A2,s2,s32));

[N2,F2] = size(B2);
Bmean2 = mean(B2); Bvar2 = var(B2); o = ones(N2,1);
b2 = (B2-o*Bmean2)./sqrt(o*Bvar2);
B2new = B2;

% kmeans algorithm giving cluster number of each row index and the mean
% vectors of each cluster
k = 4; % number of clusters
[idx2, C2] = kmeans(b2,k);
oc = ones(k,1);
% obtaining true value of colors and replacing row belonging to a cluster
% with mean vector of a respective cluster
C2r = (C2.*sqrt(oc*Bvar2))+oc*Bmean2;
for i=1:k
    rows = find(idx2 == i);
    or = ones(length(rows),1);
    B2new(rows,:) = or.*C2r(i,:);
end
B2new=floor(B2new);
A2new=reshape(uint8(B2new),s12,s22,s32);
imwrite(A2new, 'medium_risk_11new.jpg');

subplot(1,2,2);
imshow(A2new,'Border','tight');
title('After clustering with k=4')
suptitle('Medium risk mole');
print('medium_risk','-dpng')
%% High-risk mole

A3 = imread('melanoma_19.jpg');
figure
subplot(1,2,1);
imshow(A3,'Border','tight');
title('Camera image');
[s13, s23, s33]=size(A3);
s3=s13*s23;
B3=double(reshape(A3,s3,s33));

[N3,F3] = size(B3);
Bmean3 = mean(B3); Bvar3 = var(B3); o = ones(N3,1);
b3 = (B3-o*Bmean3)./sqrt(o*Bvar3);
B3new = B3;
% kmeans algorithm giving cluster number of each row index and the mean
% vectors of each cluster
k = 4; % number of clusters
[idx3, C3] = kmeans(b3,k);
oc = ones(k,1);
% obtaining true value of colors and replacing row belonging to a cluster
% with mean vector of a respective cluster
C3r = (C3.*sqrt(oc*Bvar3))+oc*Bmean3;
for i=1:k
    rows = find(idx3 == i);
    or = ones(length(rows),1);
    B3new(rows,:) = or.*C3r(i,:);
end

B3new=floor(B3new);
A3new=reshape(uint8(B3new),s13,s23,s33);
imwrite(A3new, 'melanoma_19new.jpg');

subplot(1,2,2);
imshow(A3new,'Border','tight');
title('After clustering with k=4')
suptitle('High risk mole (Melanoma)');
print('high_risk','-dpng')