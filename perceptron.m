clear all; close all;

Input = importdata('Delivery/Input.mat');
Output = importdata('Delivery/Output.mat');

% Create and train a multilayer perceptron

n_iterations = 15;

%% case 1
fnc_hidd = 'logsig';
fnc_out = 'logsig';
perf_func = 'sse';

%% case 2
% fnc_hidd = 'logsig';
% fnc_out = 'softmax';
% perf_func = 'crossentropy';

%% TWEAK
n_hidd = 50;
% n_hidd = 200;
% n_hidd = 500;

%% TWEAK A, B, C
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


accuracies = zeros(1, n_iterations);
trs = [];

for i=1:n_iterations  

    net = patternnet(n_hidd, 'traingdx');  
    
    net.divideFcn = 'divideint'; % divideFCN to allow change in data
    net.divideParam.trainRatio = p_train; % Ratio of data used as training set
    net.divideParam.valRatio = p_val;
    net.divideParam.testRatio = p_test; % Ratio of data used as test set

    net.layers{1}.transferFcn = fnc_hidd;
    net.layers{2}.transferFcn = fnc_out;

    net.performFcn = perf_func;

    net.trainParam.mc = 0.9; % momentum 
    net.trainParam.lr = 0.05; % learning rate 
    net.trainParam.max_fail = 6; % validation check 
    net.trainParam.epochs = 500; % number of epochs 
    net.trainParam.min_grad = 1e-5; % mini performance gradient

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