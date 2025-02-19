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
        % If we have found a better hypothesis, we store it
        Best.H = H;
    end
    
else
    for j = 1:prediction.n % For each Ei we iterate over all the features
        if compatibility.IC(i, j) && jointly_compatible(prediction, observations, H)
            % If Ei is compatible with Ej, we associate them and keep exploring this branch
            JCBB_R(prediction, observations, compatibility, [H j], i + 1);
        end
    end

    % Bounding: If we still have room for improvement H best, we associate Ei with a new feature
    if pairings(H) + (observations.m - i) > pairings(Best.H)
        JCBB_R(prediction, observations, compatibility, [H 0], i + 1);
    end
end

%-------------------------------------------------------
% 
%-------------------------------------------------------
function p = pairings(H)

p = length(find(H));

