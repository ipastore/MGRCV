function vargout = subplot_tight(m, n, p, margins, isWrapper, varargin)
    %% subplot_tight
    % A subplot function substitude with margins user tunabble parameter.

    %isWrapper = false;
    
    if (nargin < 4) || isempty(margins)
        % default margins value- 4% of figure
        margins = [0.04, 0.04];    
    end
    
    if length(margins) == 1
        margins(2) = margins;    
    end
    
    %note n and m are switched as Matlab indexing is column-wise, 
    % while subplot indexing is row-wise
    [subplot_col,subplot_row] = ind2sub([n, m], p);  
    height = (1 - (m + 1) * margins(1)) / m; % single subplot height
    width = (1 - (n + 1) * margins(2)) / n;  % single subplot width
    
    % note subplot suppors vector p inputs- so a merged subplot
    % of higher dimentions will be created

    % number of column elements in merged subplot 
    subplot_cols = 1 + max(subplot_col) - min(subplot_col);

    % number of row elements in merged subplot   
    subplot_rows = 1 + max(subplot_row) - min(subplot_row); 

    % merged subplot height
    merged_height = subplot_rows * (height+margins(1)) - margins(1);

    % merged subplot width
    merged_width = subplot_cols * (width +margins(2)) - margins(2);   

    % merged subplot bottom position
    merged_bottom = (m - max(subplot_row)) * (height + margins(1))...
        + margins(1); 

    % merged subplot left position
    merged_left = min(subplot_col) * (width+margins(2)) - width;              
    pos = [merged_left, merged_bottom, merged_width, merged_height];
    
    if isWrapper
        h = subplot(m, n, p, varargin{:}, ...
            'Units', 'Normalized', 'Position', pos);    
    else
        h = axes('Position', pos, varargin{:});    
    end
    
    if nargout == 1
        vargout = h;    
    end

end
