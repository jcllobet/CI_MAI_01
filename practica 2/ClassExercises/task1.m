%% 1) Define the neuron parameters
% Neuron weights
w = [4 -2];
% Neuron bias
b = -3;
% Activation function: Hyperbolic tangent sigmoid function
func = 'tansig';
% Activation function: Logistic sigmoid transfer function
% func = 'logsig'
% Activation function: Hard-limit transfer function (threshold)
% func = 'hardlim'
% Activation function: Linear transfer function
% func = 'purelin'

%% 2) Define input vectors
p = [2 3]

%% 3) Calculate neuron output
% Aggregation function
activation_potential = p*w'+b;
% Activation function
neuron_output = feval(func, activation_potential)

%% 4) Plot neuron output over the range of inputs
[p1,p2] = meshgrid(-10:.25:10);
z = feval(func, [p1(:) p2(:)]*w'+b );
z = reshape(z,length(p1),length(p2));
plot3(p1,p2,z);
grid on;
xlabel('Input 1');
ylabel('Input 2');
zlabel('Neuron output');

%% 5) Change the activation function and plot neuron output again to see the different output surfaces
%Activation/Transfer functions:
% hardlim: Positive hard limit transfer function; 
% hardlims: Symmetric hard limit transfer function; 
% purelin: Linear transfer function; 
% satlin: Positive saturating linear transfer function;
% logsig: Logistic sigmoid transfer function; 
% tansig: Hyperbolic tangent sigmoid symmetrictransfer function