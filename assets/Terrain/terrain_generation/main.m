clear all; close all; clc; debug_mode = false;
%% Terrain Generation
% This script generates a custom terrain
seed = 42;
rng(seed,"twister");

%% Terrain Size
terrain.size = 1000; % m
terrain.res  = 10;   % m/point
%% Terrain Type
% Mountain
% Forest
% City

terrain_options = "What feature do you want to add? \n m: Mountain\n w: water\n b: building\n t: trees\n c: clouds\n";

%% Terrain XYZ Flat Mesh
terrain.mesh = flat_earth(terrain);
fv_terrain = surf2patch(terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z, 'triangles');
V_all = fv_terrain.vertices;
F_all = fv_terrain.faces;

% Initialize vertex offset
v_offset = size(V_all, 1);

%% Plot the Flat Terrain for Custom Feature Setting
fig=figure(1);
ax = gca;
hold on;
f.flat = mesh(terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z);

%% Custom Feature Setting
stopFeatures = false;
while ~stopFeatures
    % Ask to add more features
    stopFeatures = strcmpi(input("Do you want to add more features? (y/n)   ", 's'),'n');
    if ~stopFeatures
        % Ask about the desired feature
        feature = input(terrain_options, 's');
        if strcmpi(feature,'m')
            PATH = fullfile(pwd,"terrain_type/Mountain/");
            % Adding a mountain
            if debug_mode
                filename = fullfile(PATH,"Mountain_5Peaks_200Height_0Seed_debug");
                [terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z] = generate_mountainous_terrain(5,200,terrain.mesh);              
            else
                height = input("What is the max height of the mountain? ");
                NumPeaks = input("How many peaks does the mountain have? ");
                [terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z] = generate_mountainous_terrain(NumPeaks,height,terrain.mesh);
                filename = "Mountain_"+num2str(NumPeaks)+"Peaks_"+num2str(height)+"Height_"+num2str(seed)+"Seed";
                filename = fullfile(PATH,filename);
            end
                figure;
                surf(terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z, 'EdgeColor', 'none');
                colormap(turbo);
                camlight;
                lighting gouraud;
                hold on;
                fv_mount = surf2patch(terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z, 'triangles');
                save_as_asc_file(terrain.mesh.Z,filename,terrain.res);
                save_as_obj_file(fv_mount.faces, fv_mount.vertices,filename,terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z);


        elseif strcmpi(feature,'w')
            % Adding a body of water

        elseif strcmpi(feature,'b')
            % Adding a building

        elseif strcmpi(feature,'t')
            % Adding a section of trees
            if debug_mode
                % Number of trees
                num_trees = 200;

                % Get terrain boundaries
                x_range = [min(terrain.mesh.X(:)), max(terrain.mesh.X(:))];
                y_range = [min(terrain.mesh.Y(:)), max(terrain.mesh.Y(:))];

                % Generate random (x,y) positions for trees
                tree_positions = [ ...
                    rand(num_trees, 1) * diff(x_range) + x_range(1), ...
                    rand(num_trees, 1) * diff(y_range) + y_range(1)
                    ];

                % Interpolate terrain height at tree positions
                tree_heights = interp2(terrain.mesh.X,terrain.mesh.Y,terrain.mesh.Z, tree_positions(:,1), tree_positions(:,2));

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
                    % Trunk
                    fv_trunk = surf2patch(Xt, Yt, Zt, 'triangles');
                    fv_trunk.faces = fv_trunk.faces + v_offset;
                    V_all = [V_all; fv_trunk.vertices];
                    F_all = [F_all; fv_trunk.faces];
                    v_offset = size(V_all, 1);
                    % Canopy
                    fv_canopy = surf2patch(Xc, Yc, Zc, 'triangles');
                    fv_canopy.faces = fv_canopy.faces + v_offset;
                    V_all = [V_all; fv_canopy.vertices];
                    F_all = [F_all; fv_canopy.faces];
                    v_offset = size(V_all, 1);
                end
            else
                disp("skip");
            end

        elseif strcmpi(feature,'c')
            % Adding a cloud

        end
    end
end



