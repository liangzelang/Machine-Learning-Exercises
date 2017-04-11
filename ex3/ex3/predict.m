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

aOne = [ones(size(X,1),1) X];  % add a(1,0)

zTwo = Theta1 * aOne'; %25*500
aTwo = sigmoid(zTwo);   %
aTwo = [ones(1, size(aTwo,2)); aTwo] ;% 26*500

zThree = Theta2 * aTwo; %10*500
aThree = sigmoid(zThree);  % h(x)

[j, i] = max(aThree', [], 2);
for k = 1:m
    p(k,1) = i(k,1);
end





% =========================================================================


end
