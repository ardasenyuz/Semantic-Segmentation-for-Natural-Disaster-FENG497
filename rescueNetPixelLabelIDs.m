function labelIDs = rescueNetPixelLabelIDs()
% Return the label IDs corresponding to each class.

labelIDs = { ...
    
    % "Background"
    [
    000 000 000; ... % "Background"
    ]
    
    % "Water" 
    [
    061 230 250; ... % "Water"
    ]
    
    % "Building_No_Damage"
    [
    180 120 120; ... % "BuildingNoDamage"
    ]
    
    % Building_Minor_Damage
    [
    235 255 007; ... % "BuildingMinorDamage"
    ]
    
    % "Building_Major_Damage"
    [
    255 184 006; ... % "BuildingMajorDamage" 
    ]
        
    % "Building_Total_Destruction"
    [
    255 000 000; ... % "BuildingTotalDestruction"
    ]
    
    % "Vehicle"
    [
    255 000 245; ... % "Vehicle"
    ]
    
    % "Road-Clear"
    [
    140 140 140; ... % "Road-Clear"
    ]
    
    % "Road-Blocked"
    [
    160 150 020; ... % "Road-Blocked"
    ]
    
    % "Tree"
    [
    004 250 007; ... % "Tree"
    ]
    
    % "Pool"
    [
    000 101 140; ... % "Pool"
    ]
    
    };
end