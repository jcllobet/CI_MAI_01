 Neuron weights
w = [4 -2];
% Neuron bias
b = -3;
% Activation function: Hyperbolic tangent sigmoid function
% func = 'tansig';
% Activation function: Logistic sigmoid transfer function
func = 'logsig';
% Activation function: Hard-limit transfer function (threshold)
% func = 'hardlim'
% Activation function: Linear transfer function
% func = 'purelin'

 
data = load('caltech101_silhouettes_28.mat');
% data
train_data = data.X.';
train_labels = data.Y;
train_classnames = data.classnames;
% 
% net = newfit(train_data,train_labels,20)
% net = train(net, train_data, train_labels)

 

 

 
%% NN from matlab site

 
% inputs = train_data;
% targets = train_labels;
% 
% hiddenLayerSize = 50;
% net = fitnet(hiddenLayerSize);
% 
% net.divideParam.trainRatio = 70/100; % Ratio of data used as training set
% net.divideParam.valRatio = 15/100; % Ratio of data used as validation set
% net.divideParam.testRatio = 15/100; % Ratio of data used as test set
% 
% [net,tr] = train(net, inputs, targets);
% 
% outputs = net(inputs);
% errors = gsubtract(targets,outputs)
% performance = perform(net, targets, outputs)
% 
% view(net)
% 

 
%% Train NN

 
inputs = train_data;
targets = train_labels;

 
hiddenLayerSize = [150];
net = patternnet(hiddenLayerSize);

 

 

 
% net = feedforwardnet(hiddenLayerSize);
net.divideFcn = 'divideint'; % divideFCN allow to change the way the data is 
                             % divided into training, validation and test data sets. 
net.divideParam.trainRatio = 0.6; % Ratio of data used as training set
net.divideParam.valRatio = 0.2; % Ratio of data used as validation set
net.divideParam.testRatio = 0.2; % Ratio of data used as test set

 
% net.trainParam.mc = 0.8; % momentum parameter 
net.trainParam.max_fail = 20; % validation check parameter

 
% net.trainParam.epochs=10; % number of epochs parameter 
net.trainParam.lr = 0.01; % learning rate parameter 
net.trainParam.min_grad = 1e-5; % minimum performance gradient
% you can define different transfer functions for each layer (layer{1} and 
% layer{2}). You can take a look to this parameter in Matlab to see all
% functions available
net.layers{1}.transferFcn = 'logsig';
% net.layers{1}.transferFcn = 'elliotsig'; 
% net.layers{1}.transferFcn = 'tansig'; 
net.layers{2}.transferFcn = 'softmax'; 
% net.layers{2}.transferFcn = 'logsig';tr

 
% you can define different performance functions. You can take a look to this 
% parameter in Matlab to see all functions available
% net.performFcn='crossentropy';
% net.performFcn='sse';
net.performFcn='mse';

 
net = configure(net,inputs,train_labels); 

 

 
view(net);

 
initial_output = net(inputs)

 
[trained_net,tr] = train(net,train_data,train_labels);
% tr.trainMask{1}

 
%% Evaluation

 
% get masks
final_test = train_data.*tr.testMask{1};
final_train = train_data.*tr.trainMask{1};
final_val = train_data.*tr.valMask{1};
% final_labels = train_labels.*tr.testMask{1};

 
% remove nan
final_test(isnan(final_test))=0;
final_train(isnan(final_train))=0;
final_val(isnan(final_val))=0;
% final_labels(isnan(final_labels))=0;

 
% % get final matrixes
% final_test = train_data.*final_test;
% final_val = train_data.*final_val;
% final_train = train_data.*final_train;
% final_labels = train_labels

 
predicted = trained_net(final_test)

 

 

 
sum(predicted) == train_labels

 

 
% classes_predict = vec2ind(outputs)
% classes_original = vec2ind(train_labels)

 
% sum(classes_predict)
% sum(classes_original)

 
% YPred = predict(trained_net,test_mask);
% 
% YPred
% % Evaluate the performance of the model by calculating the root-mean-square error (RMSE) of the predicted and actual angles of rotation.
% 
% rmse = sqrt(mean((final_test - final_val).^2))

 
% add masks to train_data
outputs = trained_net(train_data);

 
% add masks to this data too
classes_predict = vec2ind(outputs);
classes_original = vec2ind(train_labels);

 
% calculate accuracy by comparing the two matrixes