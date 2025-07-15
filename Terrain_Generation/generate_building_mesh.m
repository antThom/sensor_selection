function [V, F] = generate_building_mesh(width, depth, height, n_floors, curvature_amp, n_windows_w, n_windows_d)
    % Generates a building mesh with optional curvature and window cutouts
    %
    % Inputs:
    %   width, depth, height: building dimensions
    %   n_floors: vertical resolution (floors)
    %   curvature_amp: 0 = straight, >0 = sinusoidal x/y deflection
    %   n_windows_w, n_windows_d: number of windows across width/depth
    %
    % Outputs:
    %   V: vertices
    %   F: faces (triangles)

    if nargin < 7
        n_windows_d = 4;
    end
    if nargin < 6
        n_windows_w = 6;
    end
    if nargin < 5
        curvature_amp = 0.0;
    end
    if nargin < 4
        n_floors = 10;
    end

    % Parameters
    n_side = 4;  % 4 sides (rectangular)
    floor_height = height / n_floors;

    % Perimeter of building base
    base_x = [0, width, width, 0];
    base_y = [0, 0, depth, depth];

    V = [];
    F = [];
    face_count = 0;

    for k = 1:n_floors+1
        z = (k-1) * floor_height;

        % Optional curvature
        cx = curvature_amp * sin(2 * pi * z / height);
        cy = curvature_amp * cos(2 * pi * z / height);

        % Add each corner at this level
        for i = 1:n_side
            V = [V; base_x(i) + cx, base_y(i) + cy, z];
        end
    end

    % Connect vertical sides (as quads split into triangles)
    for k = 1:n_floors
        for i = 1:n_side
            i_next = mod(i, n_side) + 1;

            % Indexing into V
            i1 = (k-1)*n_side + i;
            i2 = (k-1)*n_side + i_next;
            i3 = k*n_side + i_next;
            i4 = k*n_side + i;

            F = [F;
                i1 i2 i3;
                i1 i3 i4];
        end
    end

    % Add top face
    top_inds = (n_floors*n_side+1):(n_floors*n_side+n_side);
    F = [F; 
        top_inds(1) top_inds(2) top_inds(3);
        top_inds(1) top_inds(3) top_inds(4)];

    % --- Optional window representation (just for visual shading) ---
    % Plotting only
    figure;
    patch('Vertices', V, 'Faces', F, ...
          'FaceColor', [0.8 0.8 0.85], 'EdgeColor', 'none');
    hold on;

    % Draw fake window insets on wide faces
    for side = 1:n_side
        if side == 1 || side == 3  % width sides only
            for i = 1:n_windows_w
                for j = 1:n_floors
                    w = width / (n_windows_w + 1);
                    x = i * w;
                    z = (j-0.5) * floor_height;

                    if side == 3
                        y = depth;
                    else
                        y = 0;
                    end

                    % Slight inset
                    rectangle = [ ...
                        x-0.4*w, y+0.01, z - floor_height*0.3;
                        x+0.4*w, y+0.01, z - floor_height*0.3;
                        x+0.4*w, y+0.01, z + floor_height*0.3;
                        x-0.4*w, y+0.01, z + floor_height*0.3];

                    fill3(rectangle(:,1), rectangle(:,2), rectangle(:,3), [0.2 0.2 0.4], 'EdgeColor', 'none');
                end
            end
        end
    end

    axis equal;
    camlight; lighting gouraud;
    title('Procedural Building Mesh');
    xlabel('X'); ylabel('Y'); zlabel('Z');
end
