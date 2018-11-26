% Neuron weights
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

data = load('caltech101_silhouettes_28_split1.mat');
data
train_data = data.train_data.';
train_labels = data.train_labels.';

net = newfit(train_data,train_labels,20)
net = train(net, train_data, train_labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TO DO
% 1) The following two configurations of the hidden and output layers transfer functions and the performance function: 
% 1.1) logsig for the hidden layer, logsig for the output layer and sum squared error as performance function; 
% 1.2) logsig for the hidden layer, softmax for the output layer and cross- entropy as performance function.
% 2) Different number of hidden units: 50, 200 and 500.
% 3) Different percentage of training, validation and test data sets: 80/10/10, 40/20/40 and 10/10/80.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






p = [2 3];


%----------------------------------------------------------------------
% test-----------------------------------------------------------------


% training-----------------------------------------------------------------
% Create a Pattern Recognition Network
fprintf('Training Neural Net\n');
hiddenLayerSize = [300];
net = patternnet(hiddenLayerSize);
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
net.performFcn='msereg';
net.performParam.ratio=0.5;
%net.trainParam.max_fail = 20;
[net,tr] = train(net,data',labels');
save 'net.mat' net;
nnoutputs = net(double(data'));
figure; plotconfusion(labels',nnoutputs);
fprintf('Training complete\n');

% commentary: For regularization use TRAINBR with defaults.
% The number of hidden nodes of H=300 is more that an order of magnitude off. Try using much smaller values.
% test-----------------------------------------------------------------
%----------------------------------------------------------------------



% Aggregation function
activation_potential = p*w'+b;
% Activation function
neuron_output = feval(func, activation_potential);

% [p1,p2] = meshgrid(-10:.25:10);
% z = feval(func, [p1(:) p2(:)]*w'+b );
% z = reshape(z,length(p1),length(p2));
% plot3(p1,p2,z);
% grid on;
% xlabel('Input 1');
% ylabel('Input 2');
% zlabel('Neuron output');

% nnd2n1;

inputs = [1:6]; % input vector (6-dimensional pattern); i.e. 1 2 3 4 5 6
outputs = [7:12]; % corresponding target output vector; i.e. 7 8 9 10 11 12

% create the network: 1 input, 2 layer (1 hidden layer and 1 output layer), feed-forward
network
net = network( ...
1, ... % numInputs (number of inputs)
2, ... % numLayers (number of layers)
[1; 0], ... % biasConnect (numLayers-by-1 Boolean vector)
[1; 0], ... % inputConnect (numLayers-by-numInputs Boolean matrix)
[0 0; 1 0], ... % layerConnect (numLayers-by-numLayers Boolean matrix); [a b; c d]
... % a: 1st-layer with itself, b: 2nd-layer with 1st-layer,
... % c: 1st-layer with 2nd-layer, d: 2nd-layer with itself
[0 1] ... % outputConnect (1-by-numLayers Boolean vector)
);
% 
% network
% net = network(1,2,[1; 0],[1; 0],[0 0; 1 0], [0 1]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST NN

net.divideFcn = 'divideint'; % divideFCN allow to change the way the data is 
                             % divided into training, validation and test data sets. 
net.divideParam.trainRatio = 0.1; % Ratio of data used as training set
net.divideParam.valRatio = 0.1; % Ratio of data used as validation set
net.divideParam.testRatio = 0.8; % Ratio of data used as test set

net.trainParam.mc = 0.8; % momentum parameter 
net.trainParam.max_fail = 6; % validation check parameter

net.trainParam.epochs=2000; % number of epochs parameter 
net.trainParam.lr = 0.01; % learning rate parameter 
net.trainParam.min_grad = 1e-5; % minimum performance gradient
% you can define different transfer functions for each layer (layer{1} and 
% layer{2}). You can take a look to this parameter in Matlab to see all
% functions available
net.layers{1}.transferFcn = 'logsig';
net.layers{1}.transferFcn = 'elliotsig'; 
net.layers{1}.transferFcn = 'tansig'; 
net.layers{2}.transferFcn = 'softmax'; 
net.layers{2}.transferFcn = 'logsig';

% you can define different performance functions. You can take a look to this 
% parameter in Matlab to see all functions available
net.performFcn='crossentropy';
net.performFcn='sse';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% View network structure
% view(net);

% number of hidden layer neurons
net.layers{1}.size = 5;
% hidden layer transfer function
net.layers{1}.transferFcn = 'logsig';
% view(net);

% % number of hidden layer neurons
% net.layers{1}.size = 5;
% % hidden layer transfer function
% net.layers{1}.transferFcn = 'logsig';
% view(net);

net = configure(net,inputs,outputs);
view(net);
