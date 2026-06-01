function setFigurePositions(cols)
    % Set Positions for All Figures
    % cols : 숫자, 열의 수

    % Get all figure handles
    fig_handles = findall(groot, 'Type', 'figure');
    fig_handles = flipud(fig_handles);  % Reverse the order of the handles

    % Number of figures
    num_figures = length(fig_handles);

    % Positioning parameters
    width  = 500;
    height = 400;
    h_margin = 10;  % Horizontal margin
    v_margin = 80;  % Vertical margin

    % ---- Added: global upward shift (pixels) ----
    y_offset = 120; % 양수면 전체 figure가 위로 올라감 (원하는 만큼 조절)

    % Compute the number of rows based on the number of figures and columns
    rows = ceil(num_figures / cols);

    % Compute positions for figures
    positions = zeros(num_figures, 4);
    for i = 1:num_figures
        row = floor((i-1) / cols);
        col = mod(i-1, cols);

        x = col*(width + h_margin);
        y = (rows-row-1)*(height + v_margin) + y_offset;

        positions(i, :) = [x, y, width, height];
    end

    % Apply positions to figures
    for i = 1:num_figures
        set(fig_handles(i), 'Position', positions(i, :));
    end
end
