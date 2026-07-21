function mountain_addition(perimeter,centers,max_heights,N,terrain)
%
X = terrain.X;
Y = terrain.Y;
Z = terrain.Z;
min_heights = max_heights * 0.1;
heights = min_heights + (max_heights-min_heights)*rand(N,1);

% Find the point nearest to the centers
for c=1:N
    [x_center,xc_Index] = min(abs(X - centers(c,1)),[],'all');
    [y_center,yc_Index] = min(abs(Y - centers(c,2)),[],'all');
    centers(c,:) = [X(xc_Index),Y(yc_Index)];
end

% Ellipsoid Function
ellipse_Z = @(x,x0,y,y0,z0,a,b,c) z0 + sqrt(c.^2.*(1 - ((x-x0)/(a)).^2 - ((y-y0)/(b)).^2));

% Make a Mountain
for ii=1:N
    [pts,width] = edgeFinder(centers(ii,:),perimeter);
    x_width = min(min(abs(centers(ii,1) - perimeter(:,1))) , (max(perimeter(:,1)) - min(perimeter(:,1)))/(N+1));
    y_width = min(min(abs(centers(ii,2) - perimeter(:,2))) , (max(perimeter(:,2)) - min(perimeter(:,2)))/(N+1));
    Z_mount = ellipse_Z(X,centers(ii,1),Y,centers(ii,2),0,x_width,y_width,heights(ii));
    Z_mount(imag(Z_mount)~=0) = 0;
    Z = Z + Z_mount;
end
figure(2); hold on; mesh(X,Y,Z); plot3(perimeter(:,1),perimeter(:,2),ones(size(perimeter,1)),'r');

% Clip Perimeter Edges
% X_perim_loc = X
end

function [pts,width] = edgeFinder(center,perimeter)
% Fix the indexing of this function
[val_x, indx] = sort(abs(center(1) - perimeter(1:end-1,1)));
[val_y, indy] = sort(abs(center(2) - perimeter(1:end-1,2)));
minx2 = val_x(1:2);
indx2 = indx(1:2);
miny2 = val_y(1:2);
indy2 = indy(1:2);

pts = closestPointOnLineSegment(center, perimeter(indx2(1),1), perimeter(indx2(2),indy2(2)));
width = abs(center-pts);
end

function closestPoint = closestPointOnLineSegment(p, a, b)
    % p: the given point [x, y]
    % a: the start point of the line segment [x, y]
    % b: the end point of the line segment [x, y]

    % Vector from a to b
    ab = b - a;
    % Vector from a to p
    ap = p - a;

    % Project vector ap onto ab, normalized by the length of ab squared
    t = dot(ap, ab) / dot(ab, ab);

    % Clamp t to the range [0, 1] to ensure the point lies on the segment
    t = max(0, min(1, t));

    % Compute the closest point on the segment
    closestPoint = a + t * ab;
end