function [X,Y,Z] = generate_mountainous_terrain(num_peaks,max_height,terrain)
    % generate_mountainous_terrain generates a mountainous terrain with a specified number of peaks
    % Usage: generate_mountainous_terrain(num_peaks)
    % Example: generate_mountainous_terrain(5)

    if nargin < 1
        num_peaks = 3; % Default number of peaks
    end

    % Terrain grid resolution
    min_height = 0.3*max_height;
%     grid_size = 200;
    X = terrain.X;
    Y = terrain.Y;
    maxLoc = max(X,[],'all');
    minLoc = min(X,[],'all');
%     [X, Y] = meshgrid(linspace(minLoc, maxLoc, grid_size), linspace(minLoc, maxLoc, grid_size));
    Z = zeros(size(X));

    % Generate random peaks
%     rng('shuffle'); % For varied results on each run
    for i = 1:num_peaks
        % Random peak position and height
        x0 = rand * (maxLoc-minLoc) - minLoc;
        y0 = rand * (maxLoc-minLoc) - minLoc;
        h = rand * max_height + min_height;      % Peak height
        sigma_x = rand * 200 + 50;
        sigma_y = rand * 200 + 50;

        % Optionally rotate the Gaussian (set theta = 0 for no rotation)
        theta = rand * 2 * pi;

        % Rotation matrix terms
        a = (cos(theta)^2)/(2*sigma_x^2) + (sin(theta)^2)/(2*sigma_y^2);
        b = -(sin(2*theta))/(4*sigma_x^2) + (sin(2*theta))/(4*sigma_y^2);
        c = (sin(theta)^2)/(2*sigma_x^2) + (cos(theta)^2)/(2*sigma_y^2);

        % Bivariate Gaussian
        Z = Z + h * exp( - ( a*(X - x0).^2 + 2*b*(X - x0).*(Y - y0) + c*(Y - y0).^2 ) );
    end

    % Add surface noise
    noise_amplitude = 0.3;
%     Z = Z + noise_amplitude * randn(size(Z));

     if nargout == 0
        figure;
        surf(X, Y, Z, 'EdgeColor', 'none');
        colormap(turbo);
        camlight headlight;
        lighting gouraud;
        axis equal;
        title(sprintf('Mountainous Terrain with %d Bivariate Peaks', num_peaks));
        xlabel('X'); ylabel('Y'); zlabel('Elevation');
    end
end
