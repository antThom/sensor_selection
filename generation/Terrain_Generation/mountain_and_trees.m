clear all; close all; clc;
% Load or generate terrain
rng(42,'twister');
num_peaks = 5;
[X, Y, Z] = generate_mountainous_terrain(num_peaks,300);

% Number of trees
num_trees = 200;

% Get terrain boundaries
x_range = [min(X(:)), max(X(:))];
y_range = [min(Y(:)), max(Y(:))];

% Generate random (x,y) positions for trees
tree_positions = [ ...
    rand(num_trees, 1) * diff(x_range) + x_range(1), ...
    rand(num_trees, 1) * diff(y_range) + y_range(1)
];

% Interpolate terrain height at tree positions
tree_heights = interp2(X, Y, Z, tree_positions(:,1), tree_positions(:,2));

% Visualize terrain
figure;
surf(X, Y, Z, 'EdgeColor', 'none');
colormap(turbo);
camlight;
lighting gouraud;
hold on;

% Plot trees
for i = 1:num_trees
    x = tree_positions(i,1);
    y = tree_positions(i,2);
    z = tree_heights(i);

    % Generate tree mesh
    [Xt, Yt, Zt, Xc, Yc, Zc] = generate_tree_mesh(20, 2, 10, 'ellipsoid');

    % Offset tree meshes to terrain
    Xt = Xt + x; Xc = Xc + x;
    Yt = Yt + y; Yc = Yc + y;
    Zt = Zt + z; Zc = Zc + z;

    % Draw trunk
    surf(Xt, Yt, Zt, 'FaceColor', [0.55 0.27 0.07], 'EdgeColor', 'none');
    % Draw canopy
    surf(Xc, Yc, Zc, 'FaceColor', [0.13 0.55 0.13], 'EdgeColor', 'none');
end

axis equal;
xlabel('X'); ylabel('Y'); zlabel('Elevation');
title('Trees on Mountainous Terrain');

% Terrain-aware lighting
camlight('headlight');    % Light follows the camera
material([0.4 0.6 0.5]);   % Ambient, diffuse, specular reflectance
lighting gouraud;         % Smooth lighting
