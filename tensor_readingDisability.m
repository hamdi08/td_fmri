function tensor_subject_rois_ts

clear; clc; close all; dbstop if error;
rng(3);

nModels = 5;
nTDC = 2;
nRankRange = 10;

acc_all = zeros(nModels,nTDC,nRankRange);

allSubjects = 16;
nSubjects = 14;
nROIs = 16;
nTimestamps = 125;

%tensor model 1

file_name_frag1 = 'subject';
file_name_frag2 = '.csv';
fmri_tensor = zeros(nSubjects, nTimestamps, nROIs);
count = 0;
for i=1:allSubjects
    if(i==8 || i==16)
        disp('Subject')
        disp(i);
        disp('is skipped');
        %continue;
    else
        file_name = string(file_name_frag1) + string(i) + string(file_name_frag2); %#ok<STRQUOT>
        disp('Processed...')
        disp(file_name);
        count = count + 1;
        disp('Tensor entry:')
        disp(count);
        fmri_tensor(count, :, :) = csvread(char(file_name), 1, 0);
    end
end
y = load('labels_14.txt')';

XX = tensor(fmri_tensor);

%TM1 -- CP

acc_T1_CP = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = cp_als(XX,R);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T1_CP(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(1);
plot([1:nRankRange], acc_T1_CP, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of CP decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T1_CP);
op_str = string('Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' with mean = ' + string(mean(acc_T1_CP));
disp(char(op_str));

acc_all(1,1,:) = acc_T1_CP;

% TM 1 -- Tucker
acc_T1_Tucker = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = tucker_als(XX,[R 2 2]);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T1_Tucker(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(2);
plot([1:nRankRange], acc_T1_Tucker, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of subject factor matrix in Tucker decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T1_Tucker);
op_str = string('In Tucker, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' with mean = ' + string(mean(acc_T1_Tucker));
disp(char(op_str));

acc_all(1,2,:) = acc_T1_Tucker;

%tensor model 2

rng(3);

load('DS2.mat');

fmri_tensor = zeros(nSubjects, nROIs, nROIs);
count = 0;
for i=1:allSubjects
    if(i==8 || i==16)
        disp('Subject')
        disp(i);
        disp('is skipped');
        %continue;
    else
        %file_name = string(file_name_frag1) + string(i) + string(file_name_frag2); %#ok<STRQUOT>
        func_mat = graph_db{i,2};
        func_mat(isnan(func_mat)) = [1];
        disp('Processed matrix...')
        disp(i);
        count = count + 1;
        disp('Tensor entry:')
        disp(count);
        fmri_tensor(count, :, :) = func_mat;
    end
end
y = load('labels_14.txt')';

XX = tensor(fmri_tensor);

%TM 2 -- Cp

acc_T2_CP = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = cp_als(XX,R);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T2_CP(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(3);
plot([1:nRankRange], acc_T2_CP, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of CP decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T2_CP);
disp('Tensor modeling: subjects, rois, rois: unthresholded correlation matrix');
op_str = string('In CP, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) +' while mean =' + string(mean(acc_T2_CP));
disp(char(op_str));

acc_all(2,1,:) = acc_T2_CP;

%TM 2 -- Tucker

%rng(3);
acc_T2_Tucker = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = tucker_als(XX,[R 2 2]);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T2_Tucker(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(4);
plot([1:nRankRange], acc_T2_Tucker, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of subject factor matrix in Tucker decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T2_Tucker);
op_str = string('In Tucker, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' while mean = ' + string(mean(acc_T2_Tucker));
disp('Tensor modeling: subjects, rois, rois: unthresholded correlation matrix');
disp(char(op_str));

acc_all(2,2,:) = acc_T2_Tucker;

%tensor model 3
rng(3);

load('DS2.mat');

fmri_tensor = zeros(nSubjects, nROIs, nROIs);
count = 0;
for i=1:allSubjects
    if(i==8 || i==16)
        disp('Subject')
        disp(i);
        disp('is skipped');
        %continue;
    else
        %file_name = string(file_name_frag1) + string(i) + string(file_name_frag2); %#ok<STRQUOT>
        func_mat = graph_db{i,3};
        %func_mat(isnan(func_mat)) = [1];
        disp('Processed matrix...')
        disp(i);
        count = count + 1;
        disp('Tensor entry:')
        disp(count);
        fmri_tensor(count, :, :) = func_mat;
    end
end
y = load('labels_14.txt')';

XX = tensor(fmri_tensor);

%TM 3 -- Cp

acc_T3_CP = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = cp_als(XX,R);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T3_CP(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(5);
plot([1:nRankRange], acc_T3_CP, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of CP decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T3_CP);
disp('Tensor modeling: subjects, rois, rois: thresholded correlation matrix');
op_str = string('In CP, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' while mean ' + string(mean(acc_T3_CP));
disp(char(op_str));
acc_all(3,1,:) = acc_T3_CP;


%TM 3 -- Tucker

acc_T3_Tucker = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = tucker_als(XX,[R 2 2]);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T3_Tucker(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(6);
plot([1:nRankRange], acc_T3_Tucker, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of subject factor matrix in Tucker decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T3_Tucker);
op_str = string('In Tucker, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' while mean = ' + string(mean(acc_T3_Tucker));
disp('Tensor modeling: subjects, rois, rois: thresholded correlation matrix');
disp(char(op_str));
acc_all(3,2,:) = acc_T3_Tucker;

%tensor model 4
rng(3);
load('DS2.mat');

fmri_tensor = zeros(nSubjects, nROIs, nROIs);
count = 0;
for i=1:allSubjects
    if(i==8 || i==16)
        disp('Subject')
        disp(i);
        disp('is skipped');
        %continue;
    else
        %file_name = string(file_name_frag1) + string(i) + string(file_name_frag2); %#ok<STRQUOT>
        func_mat = graph_db{i,2};
        func_mat(isnan(func_mat)) = [1];
        disp('Processed matrix...')
        disp(i);
        count = count + 1;
        disp('Tensor entry:')
        disp(count);
        fmri_tensor(count, :, :) = func_mat;
    end
end
y = load('labels_14.txt')';


JaccardTensor = zeros(nSubjects, nSubjects, nROIs);
for i =1:nSubjects
    for j=1:nSubjects
        mat1 = reshape(fmri_tensor(i, :, :), [nROIs, nROIs]);
        mat2 = reshape(fmri_tensor(j, :, :), [nROIs, nROIs]);
        J_vec = zeros(nROIs,1);
        for k=1:nROIs
            vec1 = mat1(k,:);
            vec2 = mat2(k,:);
            J = sum(min(vec1,vec2))/sum(max(vec1,vec2));
            J_vec(k) = J;
        end
        JaccardTensor(i,j,:) = J_vec; 
    end
end

XX = tensor(JaccardTensor);


acc_T4_CP = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = cp_als(XX,R);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T4_CP(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(7);
plot([1:nRankRange], acc_T4_CP, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of CP decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T4_CP);
disp('Tensor modeling: subjects, subjects, rois: nodewise Jaccard on unthresholded correlation matrix');
op_str = string('In CP, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' while mean = ' + string(mean(acc_T4_CP));
disp(char(op_str));
acc_all(4,1,:) = acc_T4_CP;

%TM4 -- Tucker
acc_T4_Tucker = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = tucker_als(XX,[R 2 2]);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T4_Tucker(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(8);
plot([1:nRankRange], acc_T4_Tucker, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of subject factor matrix in Tucker decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T4_Tucker);
op_str = string('In Tucker, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' while mean = ' + string(mean(acc_T4_Tucker));
disp('Tensor modeling: subjects, subjects, rois: nodewise Jaccard on unthresholded correlation matrix');
disp(char(op_str));
acc_all(4,2,:) = acc_T4_Tucker;


%tensor model 5

rng(3);

load('DS2.mat');

fmri_tensor = zeros(nSubjects, nROIs, nROIs);
count = 0;
for i=1:allSubjects
    if(i==8 || i==16)
        disp('Subject')
        disp(i);
        disp('is skipped');
        %continue;
    else
        %file_name = string(file_name_frag1) + string(i) + string(file_name_frag2); %#ok<STRQUOT>
        func_mat = graph_db{i,3};
        %func_mat(isnan(func_mat)) = [1];
        disp('Processed matrix...')
        disp(i);
        count = count + 1;
        disp('Tensor entry:')
        disp(count);
        fmri_tensor(count, :, :) = func_mat;
    end
end
y = load('labels_14.txt')';


JaccardTensor = zeros(nSubjects, nSubjects, nROIs);
for i =1:nSubjects
    for j=1:nSubjects
        mat1 = reshape(fmri_tensor(i, :, :), [nROIs, nROIs]);
        mat2 = reshape(fmri_tensor(j, :, :), [nROIs, nROIs]);
        J_vec = zeros(nROIs,1);
        for k=1:nROIs
            vec1 = mat1(k,:);
            vec2 = mat2(k,:);
            J = sum(min(vec1,vec2))/sum(max(vec1,vec2));
            J_vec(k) = J;
        end
        JaccardTensor(i,j,:) = J_vec; 
    end
end

XX = tensor(JaccardTensor);


acc_T5_CP = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = cp_als(XX,R);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T5_CP(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(9);
plot([1:nRankRange], acc_T5_CP, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of CP decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T5_CP);
disp('Tensor modeling: subjects, subjects, rois: nodewise Jaccard on thresholded correlation matrix');
op_str = string('In CP, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' with mean = ' + string(mean(acc_T5_CP));
disp(char(op_str));
acc_all(5,1,:) = acc_T5_CP;

%TM5 -- Tucker
acc_T5_Tucker = zeros(nRankRange,1);

for R=1:nRankRange
    disp('R...');
    disp(R);
    P = tucker_als(XX,[R 2 2]);
    X = P.U{1};

    k=7;

    cvFolds = crossvalind('Kfold', y, k);   %# get indices of 10-fold CV
    cp = classperf(y);                      %# init performance tracker

    for i = 1:k                                  %# for each fold
        testIdx = (cvFolds == i);                %# get indices of test instances
        trainIdx = ~testIdx;                     %# get indices training instances

        %# train an SVM model over training instances
        svmModel = fitcsvm(X(trainIdx,:), y(trainIdx), ...
        'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 

        %# test using test instances
        pred = predict(svmModel, X(testIdx,:));

        %# evaluate and update performance object
        cp = classperf(cp, pred, testIdx);
    end

    %# get accuracy
    acc = cp.CorrectRate;
    acc_T5_Tucker(R) = acc;

    %# get confusion matrix
    %# columns:actual, rows:predicted, last-row: unclassified instances
    %cp.CountingMatrix
end

figure(10);
plot([1:nRankRange], acc_T5_Tucker, '-ko', 'LineWidth', 2, 'MarkerSize',10);
xlim([1 nRankRange]);
ylim([0 1]);
xlabel('Rank of subject factor matrix in Tucker decomposition');
ylabel('Accuracy');
title('Classification accuracy with SVM and stratified 7 fold cross validation');
[val, id] = max(acc_T5_Tucker);
op_str = string('In Tucker, Best accuracy : ') + string(val) + string(' achieved with R = ') + string(id) + ' while mean =' + string(mean(acc_T5_Tucker));
disp('Tensor modeling: subjects, subjects, rois: nodewise Jaccard on thresholded correlation matrix');
disp(char(op_str));
acc_all(5,2,:) = acc_T5_Tucker;

save('acc_all.mat', 'acc_all');











