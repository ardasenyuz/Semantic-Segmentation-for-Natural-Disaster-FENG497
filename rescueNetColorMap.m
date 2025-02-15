function cmap = rescueNetColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    0 0 0   % Background
    61 230 250       % Water
    180 120 120   % BuildingNoDamage
    235 255 7    % BuildingMinorDamage
    255 184 6     % BuildingMajorDamage
    255 0 0     % BuildingTotalDestruction
    255 0 245   % Vehicle
    140 140 140     % Road-Clear
    160 150 20      % Road-Blocked
    4 250 7       % Tree
    0 101 140     % Pool
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end