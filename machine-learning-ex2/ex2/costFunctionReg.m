function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = -y.*log(sigmoid(X*theta))-(1-y).*(log(1-sigmoid(X*theta)));
J = sum(predictions)/m + (lambda/(2*m))*sum(theta(2:end) .^ 2);

%for iter = 1:1500
  %first = sigmoid(X*theta)-y;
  %
  %for i = 1:n
   %   theta(i) =theta(i) - (1/m)*sum(first .* X(:,i));
  %
 %   end;
%end;
%grad = (1/m)*(X'*(sigmoid(X*theta)-y));
for i = 1:n
  if i==1
      grad(i) = (1/m)*sum((sigmoid(X*theta)-y) .* X(:,i));
   else

      grad(i) = (1/m)*sum((sigmoid(X*theta)-y) .* X(:,i)) + (lambda/m)*theta(i);
   end

%grad = theta;
end




% =============================================================

end
