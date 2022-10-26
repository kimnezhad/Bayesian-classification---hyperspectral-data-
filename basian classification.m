clear,close all

im = imread('mainpic.tif');
im5 = im(:,:,randi([1,210],5,1));

b1 = im5(:,:,1);b2 = im5(:,:,2);b3 = im5(:,:,3);b4 = im5(:,:,4);b5 = im5(:,:,5);

% train = imread('trainbw.tif');
% test = imread('testbw.tif');

train = textread('trainac.txt');
test = textread('testac.txt');

meantr = zeros(5,4);

for i = 1:5
    b = im5(:,:,i);
    for j = 1:4
        meantr(i,j) = mean(b(train == j));
    end
end

covtr = zeros(5,5,4);
for i =1:4
    p=zeros(sum(sum(train==i)),5);
    for j = 1:5
        b = im5(:,:,j);
        p(:,j) = double(b(train == i));
        covtr(:,:,i) = cov(p);
    end
end


gm = zeros(307,307,4);

for i =1:4
    for j =1:307
        for k = 1:307
            x = double([im5(j,k,1);
                im5(j,k,2);
                im5(j,k,3);
                im5(j,k,4);
                im5(j,k,5)]);
            mu = meantr(:,i);
            covinv = inv(covtr(:,:,i));
            gm(j,k,i) = -0.5 * (x - mu)' * covinv * (x - mu) - 0.5 * log(det(covtr(:,:,i)));
        end
    end
end

[c1,c2,c3,c4] = classesml(gm);

final = gtorgb(c1,c2,c3,c4);
imshow(final),title('Classified MaxLikelihood')

[cf,overall_ac,user_ac,prod_ac] = confMatrix(test,c1,c2,c3,c4);

ge = zeros(307,307,4);

for i =1:4
    for j =1:307
        for k = 1:307
            x = double([im5(j,k,1);
                im5(j,k,2);
                im5(j,k,3);
                im5(j,k,4);
                im5(j,k,5)]);
            mu = meantr(:,i);
            ge(j,k,i) = (x - mu)' * (x - mu);
        end
    end
end

[ce1,ce2,ce3,ce4] = classesmin(ge);
finale = gtorgb(ce1,ce2,ce3,ce4);
imshow(finale),title('Classified Euclidean')

[cfe,overall_ace,user_ace,prod_ace] = confMatrix(test,ce1,ce2,ce3,ce4);


gmhd = zeros(307,307,4);

for i =1:4
    for j =1:307
        for k = 1:307
            x = double([im5(j,k,1);
                im5(j,k,2);
                im5(j,k,3);
                im5(j,k,4);
                im5(j,k,5)]);
            mu = meantr(:,i);
            covinv = inv(covtr(:,:,i));
            gmhd(j,k,i) = (x - mu)' * covinv * (x - mu);
        end
    end
end

[cmhd1,cmhd2,cmhd3,cmhd4] = classesmin(gmhd);
finale = gtorgb(cmhd1,cmhd2,cmhd3,cmhd4);
imshow(finale),title('Classified Mahalanobis')

[cfmhd,overall_acmhd,user_acmhd,prod_acmhd] = confMatrix(test,cmhd1,cmhd2,cmhd3,cmhd4);





















function [c1,c2,c3,c4] = classesml(gm)

c1 = zeros(307,307);
c2 = zeros(307,307);
c3 = zeros(307,307);
c4 = zeros(307,307);

for i =1:307
    for j = 1:307
        a = gm(i,j,:);
        a = a(:);
        [~,n] = max(a);
        if n == 1
            c1(i,j) = 1;
            c1 = logical(c1);
        elseif n == 2
            c2(i,j) =1;
            c2 = logical(c2);
        elseif n == 3
            c3(i,j) =1;
            c3 = logical(c3);
        elseif n == 4
            c4(i,j) =1;
            c4 = logical(c4);
        end
    end
end

end

function [c1,c2,c3,c4] = classesmin(gm)

c1 = zeros(307,307);
c2 = zeros(307,307);
c3 = zeros(307,307);
c4 = zeros(307,307);

for i =1:307
    for j = 1:307
        a = gm(i,j,:);
        a = a(:);
        [~,n] = min(a);
        if n == 1
            c1(i,j) = 1;
            c1 = logical(c1);
        elseif n == 2
            c2(i,j) =1;
            c2 = logical(c2);
        elseif n == 3
            c3(i,j) =1;
            c3 = logical(c3);
        elseif n == 4
            c4(i,j) =1;
            c4 = logical(c4);
        end
    end
end

end

function final = gtorgb(c1,c2,c3,c4)

bf1 = zeros(307,307);
bf2 = zeros(307,307);
bf3 = zeros(307,307);
zer = zeros(307,307);

bf2(c1) = 1;
bf3(c3) = 1;
bf1(c4) = 1;
bf1(c2) = 1;
bf2(c2) = 1;

final = cat(3,bf1,bf2,bf3);
figure
end


function [cf,overall_ac,user_ac,prod_ac] = confMatrix(test,c1,c2,c3,c4)
cf = zeros(4);

veg = test == 1;
road = test == 2;
build = test == 3;
soil = test == 4;

cf(1,1) = sum(sum(c1(veg))); %
cf(1,2) = sum(sum(c2(veg)));%roadt_vegc
cf(1,3) = sum(sum(c3(veg)));%buildt_vegc
cf(1,4) = sum(sum(c4(veg)));%soilt_vegc
cf(2,1) = sum(sum(c1(road)));%vegt_roadc
cf(2,2) = sum(sum(c2(road)));%
cf(2,3) = sum(sum(c3(road)));%buildt_roadc
cf(2,4) = sum(sum(c4(road)));%soilt_roadc
cf(3,1) = sum(sum(c1(build)));%vegt_buildc
cf(3,2) = sum(sum(c2(build)));%roadt_buildc
cf(3,3) = sum(sum(c3(build)));%
cf(3,4) = sum(sum(c4(build)));%soilt_buildc
cf(4,1) = sum(sum(c1(soil)));%vegt_soilc
cf(4,2) = sum(sum(c2(soil)));%roadt_soilc
cf(4,3) = sum(sum(c3(soil)));%build_soilc
cf(4,4) = sum(sum(c4(soil)));%



user_ac = diag(cf)./sum(cf)';
prod_ac = diag(cf)./sum(cf,2);
overall_ac = sum(diag(cf))/sum(sum(cf));
end