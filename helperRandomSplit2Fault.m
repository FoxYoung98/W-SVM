function [trainInline, trainXline, trainData, trainLabel, trainLabels, testInline, testXline, testData, testLabel, testLabels] = helperRandomSplit2Fault(percent_train_split,KnowData)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.

%% load data and set percent_train_split
Labels = KnowData.Labels;
Label = KnowData.Label;
Data = KnowData.Inputdata;
Inline = KnowData.Inline;
Xline = KnowData.Xline;
percent = percent_train_split/100;

%% set idx 0 and 1
idx0 = find(Label==0);
[N0,~] = size(idx0);
idx1 = find(Label==1);
[N1,~] = size(idx1);
%% Obtain number needed for percentage split 
num_train_0 = round(percent*N0);
num_train_1 = round(percent*N1);
rng default;
P0 = randperm(N0,num_train_0);
P1 = randperm(N1,num_train_1);    
notP0 = setdiff(1:N0,P0);
notP1 = setdiff(1:N1,P1);
%% assignment
% 0 
Inline0 = Inline(idx0,:);
Xline0 = Xline(idx0,:);
Data0 = Data(idx0,:);
Label0 = Label(idx0,:);
Labels0 = Labels(idx0,:);
% 1 
Inline1 = Inline(idx1,:);
Xline1 = Xline(idx1,:);
Data1 = Data(idx1,:);
Label1 = Label(idx1,:);
Labels1 = Labels(idx1,:);
% set train and test 0 data 
train0Inline = Inline0(P0,:);
train0Xline = Xline0(P0,:);
train0 = Data0(P0,:);
train0Label = Label0(P0);
train0Labels = Labels0(P0);
test0Inline = Inline0(notP0,:);
test0Xline = Xline0(notP0,:);
test0 = Data0(notP0,:);
test0Label = Label0(notP0);
test0Labels = Labels0(notP0);
% set train and test 1 data 
train1Inline = Inline1(P1,:);
train1Xline = Xline1(P1,:);
train1 = Data1(P1,:);
train1Label = Label1(P1);
train1Labels = Labels1(P1);
test1Inline = Inline1(notP1,:);
test1Xline = Xline1(notP1,:);
test1 = Data1(notP1,:);
test1Label = Label1(notP1);
test1Labels = Labels1(notP1);
% set train and test data
trainInline = [train0Inline;train1Inline];
trainXline = [train0Xline;train1Xline];
trainData = [train0;train1];
trainLabel = [train0Label;train1Label];
trainLabels = [train0Labels;train1Labels];
testInline = [test0Inline;test1Inline];
testXline = [test0Xline;test1Xline];
testData = [test0;test1];
testLabel = [test0Label;test1Label];
testLabels = [test0Labels;test1Labels];

