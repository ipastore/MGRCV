% HDR  stencil code
% CS 129 Computational Photography, Brown U.
%
% reads in a directory and parses out the exposure values
% files should be named like: "xxx_yyy.jpg" where
% xxx / yyy is the exposure in seconds. 

function [file_paths, exposures] = parse_files(directory)

    file_list = dir(directory);
    
    file_names = {};
    file_paths = {};
    
    k = 1;

    for i = 1:size(file_list,1)
        
        if strcmp(file_list(i).name(1:1), '.') == 0 %remove hidden files
        
            file_paths{k} = [directory file_list(i).name];
            file_names{k} = file_list(i).name;
            k = k + 1;
        
        end

    end
    
    exposures = zeros(size(file_names,2),1);
    
    i = 1; 

    for file_name = file_names

        fn = file_name{1};
        [s f] = regexp(fn, '(\d+)'); %find digits
        nominator = fn(s(1):f(1));
        denominator = fn(s(2):f(2));
        exposure = str2double(nominator) / str2double(denominator);
        exposures(i) = exposure;
        i = i + 1;

    end

    % sort by exposure
    [exposures indices] = sort(exposures);
    file_paths = file_paths(indices);

    % then inverse to get descending sort order
    exposures = exposures(end:-1:1);
    file_paths = file_paths(end:-1:1);
    
end
