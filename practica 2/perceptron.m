clear all; close all;

Input = importdata('Input.mat');
Output = importdata('Output.mat');

% Create and train a multilayer perceptron

n_iterations = 5; %number of times in which we will perfom the same task

%% case 100
fnc_hidd = 'logsig';
fnc_out = 'logsig';
perf_func = 'sse';

%% case 200
%fnc_hidd = 'logsig';
%fnc_out = 'softmax';
%perf_func = 'crossentropy';

%% TWEAK Case 10, 20, 30
n_hidd = 50;
% n_hidd = 200;
% n_hidd = 500;

%% TWEAK Case 1, 2, 3
p_train = .8;
p_val = .1;
p_test = .1;

%B
% p_train = .4;
% p_val = .2;
% p_test = .4;          

%C
% p_train = .1;
% p_val = .1;
% p_test = .8;

%test 111 ... and so on

accuracies = zeros(1, n_iterations);
trs = [];

%% DO NOT PERFORM DEEPER FOR LOOPS DUE TO UNSTABLE MATLAB IN LINUX 
for i=1:n_iterations  

    net = patternnet(n_hidd, 'traingdx');  
    
    net.divideFcn = 'divideint'; % divideFCN to allow change in data
    net.divideParam.trainRatio = p_train; % Train split
    net.divideParam.testRatio = p_test; % Test split
    net.divideParam.valRatio = p_val; % Val split

    net.layers{1}.transferFcn = fnc_hidd;
    net.layers{2}.transferFcn = fnc_out;

    net.performFcn = perf_func;

    net.trainParam.mc = 0.9; % momentum 
    net.trainParam.lr = 0.001; % learning rate 
    net.trainParam.max_fail = 6; % validation check 
    net.trainParam.epochs = 2500; % number of epochs 
    net.trainParam.min_grad = 1e-5; % minumum performance gradient

    [net,tr] = train(net,Input,Output, 'useParallel', 'yes');

    test_data = Input(:, tr.testInd);
    test_labels = Output(:, tr.testInd);

    result = net(test_data);

    [resulta, resultv] = max(result);
    [ya, yv] = max(test_labels);

    accuracy(i) = sum(yv==resultv)/length(yv);

    trs = [tr trs];

    disp(strcat('Iteration ', num2str(i), ' accuracy:' , num2str(accuracy(i))))

end