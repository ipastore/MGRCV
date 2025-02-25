function [  ] = plot_g( ...
    g_red, ...
    g_green, ...
    g_blue, ...
    save_path ...
    )

    y = (0:255);
    xlimits=[-15 5];

    clf(figure(1))
    
    figure(1)
    hold on
    subplot(2,2,1)
    plot(g_red, y, 'r-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)
    
    subplot(2,2,2)
    plot(g_green, y, 'g-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)
    
    subplot(2,2,3)
    plot(g_blue, y, 'b-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)

    subplot(2,2,4)
    plot(g_red, y, 'r-', g_green,y , 'g-', g_blue, y, 'b-');
    xlabel('log Exposure X');
    ylabel('Pixel Value Z');
    xlim(xlimits)
    hold off
    saveas(gcf,save_path)

end
