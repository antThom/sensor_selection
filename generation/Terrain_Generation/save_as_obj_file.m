function save_as_obj_file(faces, vertices, filename, X, Y, Z)
filename = filename+".obj";
% Open a file to write the OBJ data
fileID = fopen(filename, 'w');

% Write vertices to the OBJ file
% fprintf(fileID, '# Vertices\n');
for i = 1:size(vertices, 1)
    fprintf(fileID, 'v %.6f %.6f %.6f\n', vertices(i, 1), vertices(i, 2), vertices(i, 3));
end

% Compute Normals
% Initialize normals array
normals = zeros(size(vertices));
% Compute normals for each face
for i = 1:size(faces, 1)
    % Get the vertices of the current face
    v1 = vertices(faces(i, 1), :);
    v2 = vertices(faces(i, 2), :);
    v3 = vertices(faces(i, 3), :);
    
    % Compute two edge vectors
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    
    % Compute the normal using the cross product
    faceNormal = cross(edge1, edge2);
    
    % Normalize the normal vector
    faceNormal = faceNormal / norm(faceNormal);
    
    % Add the face normal to each vertex normal (for averaging)
    normals(faces(i, 1), :) = normals(faces(i, 1), :) + faceNormal;
    normals(faces(i, 2), :) = normals(faces(i, 2), :) + faceNormal;
    normals(faces(i, 3), :) = normals(faces(i, 3), :) + faceNormal;
end

% Normalize the vertex normals
normals = normals ./ vecnorm(normals, 2, 2);
for i = 1:size(normals, 1)
    fprintf(fileID, 'vn %.6f %.6f %.6f\n', normals(i, 1), normals(i, 2), normals(i, 3));
end

% Write faces to the OBJ file
% fprintf(fileID, '\n# Faces\n');
for i = 1:size(faces, 1)
    fprintf(fileID, 'f %d %d %d\n', faces(i, 1), faces(i, 2), faces(i, 3));
end

% Close the file
fclose(fileID);

disp('OBJ file has been successfully created!');



end