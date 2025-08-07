function generate_quadrotor_mesh()
close all; clc;
height = 0.5;
%% Generate a vertices for the body
% Cross
vertices.bottom_vertices = [0,0,0;-1,0,0;-1,-2,0;
                   -3,-2,0;-3,-3,0;-1,-3,0;
                   -1,-5,0;0,-5,0;0,-3,0;
                   2,-3,0;2,-2,0;0,-2,0];
vertices.top_vertices = vertices.bottom_vertices + [0,0,height];

vertices.vertices = [vertices.bottom_vertices;vertices.top_vertices];

%% Faces for the body
faces.bottom = [1, 1, 2, 3;
                2, 3,12, 1;
                3, 3, 4, 5;
                4, 5, 6, 3;
                5, 6, 7, 8;
                6, 8, 9, 6;
                7, 9,10,11;
                8,11,12, 9;
                9, 3, 6, 9;
                10,9,12, 3;];
faces.top = faces.bottom + 12;

faces.sides = [1 , 1,13, 2;
               2 ,13,14, 2;
               3 , 2,14,15;
               4 , 2,15, 3;
               5 , 3,15, 4;
               6 ,15,16, 4;
               7 , 4,16, 5;
               8 , 5,17,16;
               9 , 5, 6,17;
               10,17,18, 6;
               11, 6, 7,18;
               12,18,19, 7;
               13, 7, 8,19;
               14,19,20, 8;
               15,8 , 9,20;
               16,20,21, 9;
               17, 9,10,21;
               18,21,10,22;
               19,10,11,22;
               20,22,23,11;
               21,11,12,24;
               22,24,23,11;
               23,12, 1,13;
               24,13,24,12;];

faces.faces = [faces.bottom;faces.top;faces.sides];
faces.faces(:,1) = [];

%% Body Triangulation
% TR = delaunay(vertices.vertices);
TR = triangulation(faces.faces,vertices.vertices);
figure(); triplot(TR);
VN = TR.vertexNormal;


%% Generate Rotor Guards
% Generic Guard Outer
r_o = 1.5;
r_i = 1.0;
N = 20;
Tri_cyl = generate_rotor_guard(r_o,r_i,N,height);

%% Attach Rotor to body
Num_rotor = 4;
for ii=1:Num_rotor
    points = [TR.Points;Tri_cyl.Points];
    connectivity = [TR.ConnectivityList];
end


%% Save to obj file
write_obj_file("simple_drone.obj", TR.Points, TR.ConnectivityList, VN);

end

function TRI = removeCentralFaces(tri,r)
% Find Central points for faces
cp = incenter(tri);

% Determine if center points are within a radius
center_mask = vecnorm(cp(:,1:2),2,2) < r;

% Delete All Faces in the center
face = tri.ConnectivityList(~center_mask,:);
vert = tri.Points;

TRI = triangulation(face,vert);
end

function Tri_cyl = generate_rotor_guard(r_o,r_i,N,height)
% Outer Guard
[X_o, Y_o, Z_o] = cylinder(r_o, N);
Z_o = Z_o*height;
Cyl_o = surf2patch(X_o,Y_o,Z_o);


% Generic Guard Inner
[X_i, Y_i, Z_i] = cylinder(r_i, N);
Z_i = Z_i*height;
Cyl_i = surf2patch(X_i,Y_i,Z_i);

tri = delaunayTriangulation([Cyl_o.vertices(:,1);Cyl_i.vertices(:,1)], [Cyl_o.vertices(:,2);Cyl_i.vertices(:,2)], [Cyl_o.vertices(:,3);Cyl_i.vertices(:,3)]);

% figure(3);tetramesh(tri);

% Remove all central faces
Tri_cyl = removeCentralFaces(tri,r_i);
figure(); tetramesh(Tri_cyl)
end
