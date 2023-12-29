%% Load Files
load('SeismicData.mat');

%% WaveletTransform

[SeismicData.wt,SeismicData.mra,SeismicData.reseismicdata] = helperWavelet4Data(SeismicData.Data,'db4',[true(1,6)]);

%% Input data selection

SeismicData.Inputdata = SeismicData.mra(:,:,4);

%% normalization

dataset = SeismicData.Inputdata; 
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';
SeismicData.Inputdata = dataset_scale;

%% Create Training and Test Data
percent_train = 70;
[TrainInline,TrainXline,TrainData,TrainLabel,TrainLabels,TestInline,TestXline,TestData,TestLabel,TestLabels] = helperRandomSplit2Fault(percent_train,SeismicData);
% By design the training data contains 70% of the data.
% Recall that the 0 class represents 96.5% of the data, the 1 class represents 3.6%. 
% Examine the percentage of each class in the training and test sets. 
% The percentages in each are consistent with the overall class percentages in the data set.
Ctrain = countcats(categorical(TrainLabels))./numel(TrainLabels).*100;
Ctest = countcats(categorical(TestLabels))./numel(TestLabels).*100;

%% Particle swarm optimization

% Particle Swarm Optimization Algorithm Selects the Best SVM Parameter c&g
tic
[bestacc,bestc,bestg] = psoSVMcgForClass(TrainLabel,TrainData);
toc

% Print selection result  
disp('Print selection result');
str = sprintf( 'Best Cross Validation Accuracy = %g%% Best c = %g Best g = %g',bestacc,bestc,bestg);
disp(str);

%% SVM network training with optimal parameters

cmd = [' -c ',num2str(bestc),' -g ',num2str(bestg),' -b 1 '];
model = svmtrain(TrainLabel,TrainData,cmd);

%% test

[test_label, accuracy, decision_values] = svmpredict(TestLabel, TestData, model, '-b 1');

% Testlabel is known; test_label is calculated by machine learning
C = confusionmat(TestLabel, test_label);
figure
cm = confusionchart(TestLabel, test_label);
TP1 = C(2,2); % Number of faults predicted correctly
FP1 = C(1,2); % Number of faults predicted in error
TN0 = C(1,1); % Number of non-faults predicted correctly
FN0 = C(2,1); % Number of non-faults predicted in error
Accuracy = (TP1+TN0)/(TP1+TN0+FP1+FN0);
Precision = TP1/(TP1+FP1);
Recall = TP1/(TP1+FN0);
F1_score = 2*Precision*Recall/(Precision+Recall);
