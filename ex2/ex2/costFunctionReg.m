function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% %  求取CostFunction
n = length(theta);
tmp=0;
for i = 1:m
    tmp = sigmoid(X(i,:)*theta);
    J = J + ((-y(i,:))*log(tmp)-(1-y(i,:))*log(1-tmp));
end
J = J / m;
thetaOne = theta((2:end),:);
temp = sum(thetaOne.^2)*lambda/(2*m);
J = J + temp;


sumTemp = 0;
tempTwo = 0;
%  求取偏导数
for j = 1:m
   tempTwo = sigmoid(X(j,:)*theta);
   sumTemp = sumTemp + (tempTwo - y(j,:))*X(j,1);
end
grad(1) = sumTemp/m;


for k = 2:n 
    tempOne = 0;
    sumTemp = 0;
    for j = 1:m
       tempOne = sigmoid(X(j,:)*theta);
       sumTemp = sumTemp +  (tempOne - y(j,:))*X(j,k) ;
    end
    grad(k) = sumTemp/m + lambda/m*theta(k);
end

% 
% n=length(theta);
% 
% for i=1:m
%     tmp=sigmoid(X(i,:)*theta);
%     J=J+(-y(i)*log(tmp)-(1-y(i))*log(1-tmp));
% end
% J=J/m;
% tmp=0;
% for j=2:n
%     tmp=tmp+theta(j)*theta(j);
% end
% J=J+tmp*lambda/2/m;
% 
% for j=1:n
%     tmp=0;
%     for i=1:m
%         tmp=tmp+(sigmoid(X(i,:)*theta)-y(i))*X(i,j);
%     end
%     if j==1
%     grad(j)=1/m*tmp;
%     else 
%     grad(j)=1/m*tmp+lambda/m*theta(j);        
%     end
% end




% =============================================================

end
