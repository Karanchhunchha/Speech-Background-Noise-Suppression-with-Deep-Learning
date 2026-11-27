function rootDir = setup()
%SETUP Add the project folders to the MATLAB path.
%   rootDir = SETUP() adds the project root and everything under src/
%   to the path and returns the project root directory. Every entry
%   point calls SETUP first, so the project runs from a fresh clone
%   with no manual path configuration ("one-click run").

    rootDir = fileparts(mfilename('fullpath'));
    addpath(rootDir);
    addpath(genpath(fullfile(rootDir, 'src')));
    toolsDir = fullfile(rootDir, 'tools');
    if isfolder(toolsDir)
        addpath(toolsDir);
    end
end
