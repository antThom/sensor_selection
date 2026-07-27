function generate_city_block(grid_rows, grid_cols, spacing)
    % Generates a grid of procedural buildings (city block)
    % spacing: distance between buildings

    if nargin < 3
        spacing = 6;  % default spacing
    end

    % Base building size
    base_width = 4;
    base_depth = 3;

    V_all = [];
    F_all = [];
    v_offset = 0;

    % Loop over grid
    for r = 1:grid_rows
        for c = 1:grid_cols
            % Random building features
            height = 6 + randi(6);               % random height: 6–12
            n_floors = randi([6, 12]);
            curvature = rand() * 0.5;            % curvature 0–0.5
            nw = randi([3, 6]);                  % windows width
            nd = randi([3, 5]);                  % windows depth

            % Generate mesh
            [V, F] = generate_building_mesh(base_width, base_depth, height, n_floors, curvature, nw, nd);

            % Offset building position in X and Y
            offset = [ ...
                (c-1) * spacing, ...
                (r-1) * spacing, ...
                0];
            V = V + offset;

            % Append to full mesh
            F = F + v_offset;
            V_all = [V_all; V];
            F_all = [F_all; F];
            v_offset = size(V_all, 1);
        end
    end

    % Display full city block
    figure;
    patch('Vertices', V_all, 'Faces', F_all, ...
          'FaceColor', [0.85 0.85 0.9], ...
          'EdgeColor', 'none');
    axis equal;
    camlight;
    lighting gouraud;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('City Block (%dx%d)', grid_rows, grid_cols));

    % Export to PLY
%     TR = triangulation(F_all, V_all);
%     plywrite_ascii('city_block.ply', TR);
%     fprintf('✅ Exported city_block.ply\n');
end
