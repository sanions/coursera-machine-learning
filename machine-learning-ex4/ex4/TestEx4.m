clear ; close all; clc%% Setup the parameters you will use for this exerciseinput_layer_size  = 400;  % 20x20 Input Images of Digitshidden_layer_size = 25;   % 25 hidden unitsnum_labels = 10;          % 10 labels, from 1 to 10                             % (note that we have mapped "0" to label 10)%% ================ Part 2: Loading Parameters ================% In this part of the exercise, we load some pre-initialized % neural network parameters.fprintf('\nLoading Saved Neural Network Parameters ...\n')% Load the weights into variables Theta1 and Theta2load('ex4weights.mat');% Unroll parameters nn_params = [Theta1(:) ; Theta2(:)];load('ex4data1.mat');%% =============== Part 4: Implement Regularization ===============%  Once your cost function implementation is correct, you should now%  continue to implement the regularization with the cost.%fprintf('\nChecking Cost Function (w/ Regularization) ... \n')% Weight regularization parameter (we set this to 1 here).lambda = 1;J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...                   num_labels, X, y, lambda);fprintf(['Cost at parameters (loaded from ex4weights): %f '...         '\n(this value should be about 0.383770)\n'], J);fprintf('Program paused. Press enter to continue.\n');pause;%% ================ Part 5: Sigmoid Gradient  ================%  Before you start implementing the neural network, you will first%  implement the gradient for the sigmoid function. You should complete the%  code in the sigmoidGradient.m file.%fprintf('\nEvaluating sigmoid gradient...\n')g = sigmoidGradient([-1 -0.5 0 0.5 1]);fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');fprintf('%f ', g);fprintf('\n\n');fprintf('Program paused. Press enter to continue.\n');pause;%% ================ Part 6: Initializing Pameters ================%  In this part of the exercise, you will be starting to implment a two%  layer neural network that classifies digits. You will start by%  implementing a function to initialize the weights of the neural network%  (randInitializeWeights.m)fprintf('\nInitializing Neural Network Parameters ...\n')initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);% Unroll parametersinitial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];fprintf('\nChecking Backpropagation... \n');%  Check gradients by running checkNNGradientscheckNNGradients;