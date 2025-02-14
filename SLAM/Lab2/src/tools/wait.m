function wait(message),
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------

if nargin == 0
    message = 'Press RETURN to continue...';
end

disp(message);
pause;