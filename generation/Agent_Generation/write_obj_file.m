function write_obj_file(filename, vertices, faces, normals)
fid = fopen(filename, 'w');

% Vertex lines
vertex_lines = sprintf('v %.4f %.4f %.4f\n', vertices');
fwrite(fid, vertex_lines);

% Normal lines
normal_lines = sprintf('vn %.4f %.4f %.4f\n', normals');
fwrite(fid, normal_lines);

% Face lines (with normal indices)
face_format = 'f %d//%d %d//%d %d//%d\n';
face_data = [faces, faces]';  % each vertex index used as its normal index
face_lines = sprintf(face_format, face_data);
fwrite(fid, face_lines);

fclose(fid);
end