function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


% ========== Liangzelang Code Begin ================%  

%theta(1,1) = theta(1,1) - alpha*(1/m)*sum((X*theta-y).*X(:,1));
%theta(2,1) = theta(2,1) - alpha*(1/m)*sum((X*theta-y).*X(:,2));
%theta(3,1) = theta(3,1) - alpha*(1/m)*sum((X*theta-y).*X(:,3));
 
% ========== Liangzelang Code End   ================%   
tmp_thetap=zeros(size(X,2),1);
    for i=1:size(X,2)
      tmp_thetap(i,1)=theta(i,1)-alpha*(1/m)*sum((X*theta-y).*X(:,i));
    end
    for j=1:size(X,2)
      theta(j,1)=tmp_thetap(j,1);
    end








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
