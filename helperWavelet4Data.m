function [wt,mra,redata] = helperWavelet4Data(data,wavelet,levelForReconstruction) 

[m,~] = size(data);

for i = 1:m
    %levelForReconstruction = [true(1,num)]; %num=level+approx
    %wavelet = 'wavelet';
    %w = modwt(x) returns the maximal overlap discrete wavelet transform (MODWT) of the 1-D real-valued signal, x.
    wt(:,:,i) = modwt(data(i,:), wavelet, length(levelForReconstruction)-1); 
    %mra = modwtmra(w) returns the multiresolution analysis (MRA) of the maximal overlap discrete wavelet transform (MODWT) matrix, w. 
    %The MODWT matrix, w, is the output of the modwt function.
    mra(:,:,i) = modwtmra(wt(:,:,i), wavelet); 
    redata(i,:) = sum(mra(levelForReconstruction,:,i),1);
end

wt = permute(wt,[3 2 1]);
mra = permute(mra,[3 2 1]);
