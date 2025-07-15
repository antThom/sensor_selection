function generate_quadrotor_mesh(param)
% Generates a simple quadrotor mesh and visualizes it in 3D

% Define the dimensions of the rectangular prism
param.body_length = 0.5; % Length along x-axis
param.body_width = 0.3;  % Width along y-axis
param.body_height = 0.2; % Height along z-axis

% Define the vertices of the rectangular prism
vertices = [
    0, 0, 0; % Bottom face
    length, 0, 0;
    length, width, 0;
    0, width, 0;
    0, 0, height; % Top face
    length, 0, height;
    length, width, height;
    0, width, height
    ];

% Define the faces of the rectangular prism
faces = [
    1, 2, 3, 4; % Bottom face
    5, 6, 7, 8; % Top face
    1, 2, 6, 5; % Side face
    2, 3, 7, 6; % Side face
    3, 4, 8, 7; % Side face
    4, 1, 5, 8  % Side face
    ];

% Plot the rectangular prism
figure;
patch('Vertices', vertices, 'Faces', faces, ...
    'FaceColor', 'cyan', 'EdgeColor', 'black', 'FaceAlpha', 0.5);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Rectangular Prism');
axis equal;
grid on;

% Parameters
arm_length = 0.3;
arm_radius = 0.01;
body_radius = 0.04;
body_height = 0.02;
motor_radius = 0.02;
motor_height = 0.01;

% Create central body (cylinder)
[Xb, Yb, Zb] = cylinder(body_radius, 20);
Zb = Zb * body_height - body_height/2;

% Plot body
figure;
hold on;
surf(Xb, Yb, Zb, 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none');

% Define arm orientations
angles = [0, pi/2, pi, 3*pi/2];

% Create arms and motors
for i = 1:4
    theta = angles(i);
    dx = arm_length * cos(theta);
    dy = arm_length * sin(theta);

    % Create arm along X axis, then rotate
    [Xa, Ya, Za] = cylinder(arm_radius, 12);
    Xa = Xa * 2*arm_length - arm_length;  % center arm at origin
    Za = Za * 1 - 0.005; % slight vertical offset
    % Rotate
    Xa_rot = Xa * cos(theta) - Ya * sin(theta);
    Ya_rot = Xa * sin(theta) + Ya * cos(theta);

    % Plot arm
    surf(Xa_rot, Ya_rot, Za, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'none');

    % Motor position
    xm = dx;
    ym = dy;

    % Create motor
    [Xm, Ym, Zm] = cylinder(motor_radius, 20);
    Zm = Zm * motor_height + 0.005;

    Xm = Xm + xm;
    Ym = Ym + ym;

    surf(Xm, Ym, Zm, 'FaceColor', [0.2 0.2 0.6], 'EdgeColor', 'none');
end

% Aesthetics
axis equal;
camlight headlight;
lighting gouraud;
view(3);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Quadrotor Drone Mesh');
hold off;
end
