function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,

    tmpTheta = theta;
    for j = 1:length(theta),
	derivative = 0;
	for i = 1:m,
	    derivative = derivative + (X(i,:)*theta - y(i))*X(i,j);
	end;
	tmpTheta(j) = tmpTheta(j) - alpha*(derivative/m);
    end;
    theta = tmpTheta; 

    J_history(iter) = computeCost(X, y, theta);
    
end;

end
