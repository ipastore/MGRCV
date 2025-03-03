function [  ] = plot_g( ...
    g_red, ...
    g_green, ...
    g_blue, ...
    save_path, ...
    title_text ...
    )

    % If title_text parameter is not provided, use empty string
    if nargin < 5
        title_text = '';
    end

    y = (0:255);
    xlimits=[-15 5];

    % Create figure without displaying it
    fig = figure('visible', 'off');
    
    hold on
    subplot(2,2,1)
    plot(g_red, y, 'r-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)
    title('Red Channel');
    
    subplot(2,2,2)
    plot(g_green, y, 'g-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)
    title('Green Channel');
    
    subplot(2,2,3)
    plot(g_blue, y, 'b-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)
    title('Blue Channel');

    subplot(2,2,4)
    plot(g_red, y, 'r-', g_green,y , 'g-', g_blue, y, 'b-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)
    title('All Channels');
    
    % Add overall title if provided
    if ~isempty(title_text)
        sgtitle(title_text, 'FontSize', 14);
    end
    
    hold off
    saveas(gcf,save_path)

end
