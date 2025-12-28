function save_as_asc_file(elevation,filename,cellsize)
filename = filename+".asc";
[nrows, ncols] = size(elevation);
nodata = -9999;
fid = fopen(filename, 'w');
fprintf(fid, 'ncols         %d\n', ncols);
fprintf(fid, 'nrows         %d\n', nrows);
fprintf(fid, 'xllcorner     %.3f\n', 0);
fprintf(fid, 'yllcorner     %.3f\n', 0);
fprintf(fid, 'cellsize      %.3f\n', cellsize);
fprintf(fid, 'NODATA_value  %.1f\n', nodata);
for i = 1:nrows
    fprintf(fid, '%.3f ', elevation(i, :));
    fprintf(fid, '\n');
end
fclose(fid);
disp('ASC file has been successfully created!');
end