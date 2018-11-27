%% Task 5 - Classification of linearly separable data with a perceptron

% Objective: Two clusters of data, belonging to two classes, are defined in a 2-dimensional input
% space. Classes are linearly separable. The task is to construct a Perceptron for the classification
% of data.

% Recall: The simplest kind of neural network is a single-layer perceptron network, which consists
% of a single layer of output nodes; the inputs are fed directly to the outputs via a series of weights.
% In this way it can be considered the simplest kind of feed-forward network. Perceptrons can be
% trained by a simple learning algorithm that is usually called the delta rule. It calculates the errors
% between calculated output and sample output data, and uses this to create an adjustment to the
% weights, thus implementing a form of gradient descent.

%% 1) Define input and output data
% number of samples of each class
N = 20;

% define inputs and outputs
offset = 5; % offset for second class
x = [randn(2,N) randn(2,N)+offset]; % inputs
y = [zeros(1,N) ones(1,N)]; % outputs

% Plot input samples with plotpv (Plot perceptron input/target vectors)
figure(1)
plotpv(x,y);

%% 2) Create and train the perceptron
net = perceptron;
net = train(net, x, y);
view(net);

%% 3) Plot decision boundary
figure(1)
plotpc(net.IW{1},net.b{1});
% Plot a classification line on a perceptron vector plot