function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% temp = 0;
% for i = 1:m
%     temp = temp + (-y(i,:)*log(sigmoid(X(i,:)*theta))-(1-y(i,:))*log(1-sigmoid(X(i,:)*theta)));
% end
% tempTheta = sum(theta.^2)*lambda/(2*m);
% J = temp/m + tempTheta;
% 
% for j = 2:n
%    grad(j) = sum((sigmoid(X*theta)-y).* X(:,j))/m + lambda/m*theta(j);  
% end
% grad(1) = 1/m*sum((sigmoid(X*theta)-y).*X(:,1));


% sig=sigmoid(X*theta);
% J=1/m*sum(-log(sig).*y+log(ones(m,1)-sigmoid(X*theta)).*y-log(ones(m,1)-sigmoid(X*theta)))+lambda/(2*m)*sum((theta([2:size(theta,1)],:)).^2);
% tmp=theta;
% tmp(1)=0;
% grad=grad+1/m*(X'*(sig-y))+lambda/m*tmp;
sig = sigmoid(X*theta);
%J = 1/m*(sum(-log(sig - y).*y-log(ones(m,1)-log(sig - y)).*(ones(m,1)-y))) + lambda/(2*m)*sum((theta([2:size(theta,1)],:)).^2);
J = 1/m*(sum(-log(sig).*y-log(ones(m,1)-sig).*(ones(m,1)-y))) + lambda/(2*m)*sum((theta([2:size(theta,1)],:)).^2);
%J=1/m*sum(-log(sig).*y+log(ones(m,1)-sigmoid(X*theta)).*y-log(ones(m,1)-sigmoid(X*theta)))+lambda/(2*m)*sum((theta([2:size(theta,1)],:)).^2);
tmp = theta;
tmp(1) = 0;
grad = grad + 1/m*(X'*(sig-y)) + 1/m*lambda*tmp;

% =============================================================

grad = grad(:);

end
