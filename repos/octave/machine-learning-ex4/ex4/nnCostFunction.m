function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Make the y vectors to be matrix
y_matrix = eye(num_labels)(y,:);
% set the value of a1, which equals to X adding by 1 colum;
a1 = [ones(m, 1), X]; % a1 is m * 401
% z2 is a m*hidden_layer_size, Theta1 with size hidden_layer_size * (input_layer_size + 1)--401
z2 = a1 * Theta1';  % z2 is m * hidden_layer_size
a2 = sigmoid(z2);
a2 = [ones(m,1), a2];   % add bias column, now a2 is m * (hidden_layer_size + 1)
z3 = a2 * Theta2';  % Theta2 is size of 10 * (hidden_layer_size + 1)
a3 = sigmoid(z3);   % a3 is m * 10
J = (-1/m)*sum(sum( (y_matrix .* log(a3)) + ((1-y_matrix) .* log(1-a3)))); % y is m*10, a3 is also m * 10, a3 = h(x) 

% for the reason why can't use the matrix multiply, please check with 
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA
% for m*n, not always A'*B  equals to the sum(sum(A .* B))
Theta1_1 = Theta1(:,2:end);
%display(size(Theta1_1));
%Theta1_1(:,1) = 0;
Theta2_1 = Theta2(:,2:end);
%display(size(Theta2_1));
J = J + (lambda/(2 * m)) * (sum(sum(Theta1_1 .^ 2)) + sum(sum(Theta2_1 .^2)));

% caculate the backpropagation grad
% the deta3
deta3 = a3 - y_matrix; % a3 is m * 10, y matrix is m*10, so deta3 is m*10
deta2 = (deta3 * Theta2_1) .* sigmoidGradient(z2); % a3 is m*10, Theta2_1 is 10 * h,  z2 is m*h, deta2 is m*h
D1 = deta2' * a1; % deta2 is m*h, a1 is m*(n+1), D1 is h*(n+1)
D2 = deta3' * a2; % deta3 is m*10, a2 is m*(h+1), D2 is 10*(h+1)
Theta1_grad = (1/m) * D1; %Theta1_grad is h*(n+1)
Theta2_grad = (1/m) * D2; %Theta2_grad is r*(h+1)
grad = [Theta1_grad(:); Theta2_grad(:)];

% add regulation
Theta1_2 = Theta1;
Theta1_2(:, 1) = 0;
Theta2_2 = Theta2;
Theta2_2(:, 1) = 0;
Theta1_grad = Theta1_grad + (lambda/m)*Theta1_2;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2_2;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
