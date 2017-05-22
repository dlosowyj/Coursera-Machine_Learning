function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add a column of 1's to X for the bias
X = [ones(m,1) X];

% Calculate the first hidden layer
z_2 = X*Theta1';
a_2 = sigmoid(z_2);

% Add a column of 1's to a_2 for the bias
a_2 = [ones(m,1) a_2];
fprintf('%ix%i\n', size(a_2,1),size(a_2,2));

% Calculate the output
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);

% Get the maximum probability and index at each row
[max_probs, i] = max(a_3,[],2);

p = i;

% =========================================================================


end
