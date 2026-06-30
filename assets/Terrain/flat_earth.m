function MESH = flat_earth(terrain)
% This function recieves the size of the terrain and returns a mesh of the
% x and y data

NumPoints = terrain.size/terrain.res + 1;

x = linspace(0,terrain.size,NumPoints);
y = x;
z = zeros(size(x));

[MESH.X,MESH.Y] = meshgrid(x,y);
MESH.Z = meshgrid(z);

end