%-------------------------------------------------------
function H = JCBB (prediction, observations, compatibility)
% 
%-------------------------------------------------------
global Best;
global configuration;

Best.H = zeros(1, observations.m);

JCBB_R (prediction, observations, compatibility, [], 1);

H = Best.H;
configuration.name = 'JCBB';

%-------------------------------------------------------
function JCBB_R (prediction, observations, compatibility, H, i)
% 
%-------------------------------------------------------
global Best;
global configuration;

if i > observations.m % leaf node?
    if pairings(H) > pairings(Best.H) % did better?
        Best.H = H;
    end
else
    % complete JCBB here
end

%-------------------------------------------------------
% 
%-------------------------------------------------------
function p = pairings(H)

p = length(find(H));