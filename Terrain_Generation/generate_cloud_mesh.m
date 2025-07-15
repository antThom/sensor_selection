function [V, F] = generate_cloud_mesh(cloud_size, resolution, occupancy_threshold, seed)
    % Generates a 3D mesh of a volumetric cloud
    % Inputs:
    %   cloud_size: scalar, spatial size of the cloud volume
    %   resolution: scalar, grid resolution per axis
    %   occupancy_threshold: value between 0 and 1 (higher = denser)
    %   seed: optional RNG seed for reproducibility
    %
    % Outputs:
    %   V: Nx3 array of mesh vertices
    %   F: Mx3 array of triangle face indices

    if nargin < 4
        seed = randi(1000);
    end

    rng(seed);  % for reproducibility
    n = resolution;
    L = cloud_size;

    % 3D grid
    [X, Y, Z] = meshgrid(linspace(-L, L, n), ...
                         linspace(-L, L, n), ...
                         linspace(-L, L, n));

    % Generate base Gaussian falloff (to keep it cloud-like)
    R = sqrt(X.^2 + Y.^2 + Z.^2);
    gaussian_falloff = exp(-(R / (0.5 * L)).^2);

    % Add random noise
    noise_field = randn(size(X));
    volume_data = gaussian_falloff + 0.5 * noise_field;

    % Normalize to [0, 1]
    volume_data = (volume_data - min(volume_data(:))) / ...
                  (max(volume_data(:)) - min(volume_data(:)));

    % Extract surface using isosurface (at the occupancy threshold)
    fv = isosurface(X, Y, Z, volume_data, occupancy_threshold);

    % Smooth the result (optional post-processing)
    [V, F] = deal(fv.vertices, fv.faces);
    
    % Plot the cloud mesh
    figure;
    patch('Vertices', V, 'Faces', F, ...
          'FaceColor', [1 1 1]*0.9, ...
          'EdgeColor', 'none', ...
          'FaceAlpha', 0.6);
    camlight;
    lighting gouraud;
    axis equal off;
    title(sprintf('Cloud Mesh (threshold = %.2f, seed = %d)', occupancy_threshold, seed));
end
