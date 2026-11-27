function pairs = listPairs(cleanDir, noisyDir)
%LISTPAIRS Matched clean/noisy file pairs from two folders.
%   pairs = LISTPAIRS(cleanDir, noisyDir) returns a struct array with
%   fields id, cleanFile and noisyFile for every .wav file whose name
%   appears in BOTH folders (VoiceBank-DEMAND pairs share file names),
%   sorted by file name.

    cleanListing = dir(fullfile(cleanDir, '*.wav'));
    cleanNames = {cleanListing.name};
    noisyNames = {dir(fullfile(noisyDir, '*.wav')).name};
    matched = cleanNames(ismember(cleanNames, noisyNames));
    pairs = struct('id', cell(1, numel(matched)), ...
                   'cleanFile', cell(1, numel(matched)), ...
                   'noisyFile', cell(1, numel(matched)));
    for i = 1:numel(matched)
        pairs(i).id = matched{i}(1:end - 4);
        pairs(i).cleanFile = fullfile(cleanDir, matched{i});
        pairs(i).noisyFile = fullfile(noisyDir, matched{i});
    end
end
