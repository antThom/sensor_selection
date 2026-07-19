function [Xt, Yt, Zt, Xc, Yc, Zc] = generate_tree_mesh(trunk_height, trunk_radius, canopy_radius, canopy_type)
    % Generates a simple tree mesh: trunk (cylinder) + canopy (sphere/ellipsoid)
    if nargin < 4
        canopy_type = 'sphere';
    end

    % Trunk: vertical cylinder
    [Xt, Yt, Zt] = cylinder(trunk_radius, 20);
    Zt = Zt * trunk_height;

    % Canopy: sphere or ellipsoid
    [Xc, Yc, Zc] = sphere(20);
    Xc = Xc * canopy_radius;
    Yc = Yc * canopy_radius;

    if strcmpi(canopy_type, 'ellipsoid')
        Zc = Zc * canopy_radius * 1.5;
    else
        Zc = Zc * canopy_radius;
    end

    % Shift canopy above the trunk
    Zc = Zc + trunk_height;
end
