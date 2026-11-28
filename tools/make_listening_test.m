function keyFile = make_listening_test(varargin)
%MAKE_LISTENING_TEST Build a blinded, randomised listening-test set.
%   keyFile = MAKE_LISTENING_TEST() takes the first 'NumPairs' pairs of
%   the committed sample (data/sample), enhances each with every method
%   (noisy / spectral subtraction / Wiener / CNN), shuffles the clips
%   under blind names (clip_01.wav, ...) and writes, into 'OutDir'
%   (default 'listening_test'):
%     - the blinded clips,
%     - answer_key.csv      (clip -> source file + method; keep private
%                            until ratings are collected),
%     - rating_sheet.csv    (clip, listener, rating_1_to_5 — the file
%                            listeners fill in; ratings are NEVER
%                            pre-filled by this or any other script).
%   The seed is fixed ('Seed', default 193) so the set is reproducible.

    p = inputParser;
    p.addParameter('NumPairs', 5, @(x) isscalar(x) && x >= 1);
    p.addParameter('OutDir', 'listening_test', @ischar);
    p.addParameter('Seed', 193, @(x) isscalar(x));
    p.parse(varargin{:});

    rootDir = setup();
    outDir = fullfile(rootDir, p.Results.OutDir);
    if ~exist(outDir, 'dir'); mkdir(outDir); end
    pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                      fullfile(rootDir, 'data', 'sample', 'noisy'));
    if isempty(pairs)
        error('make_listening_test:NoData', 'No sample pairs found.');
    end
    pairs = pairs(1:min(p.Results.NumPairs, numel(pairs)));
    model = loadDenoiseModel(fullfile(rootDir, 'models', 'denoiseNet.mat'));
    methodNames = {'Noisy', 'SpectralSubtraction', 'Wiener', 'CNN'};

    clips = {};
    for i = 1:numel(pairs)
        [clean, fs] = audioread(pairs(i).cleanFile); %#ok<NASGU>
        [noisy, ~] = audioread(pairs(i).noisyFile);
        noisy = noisy(:, 1);
        enh = {noisy, spectralSubtraction(noisy, fs), ...
               wienerEnhance(noisy, fs), denoiseWithNet(noisy, fs, model)};
        for m = 1:numel(methodNames)
            clips(end + 1, :) = {pairs(i).id, methodNames{m}, enh{m}, fs}; %#ok<AGROW>
        end
    end
    rng(p.Results.Seed);
    order = randperm(size(clips, 1));
    clips = clips(order, :);

    key = table();
    sheet = table();
    for k = 1:size(clips, 1)
        clipName = sprintf('clip_%02d.wav', k);
        audiowrite(fullfile(outDir, clipName), clips{k, 3}, clips{k, 4});
        key.Clip(k, 1) = string(clipName);
        key.SourceFile(k, 1) = string(clips{k, 1});
        key.Method(k, 1) = string(clips{k, 2});
        sheet.Clip(k, 1) = string(clipName);
        sheet.Listener(k, 1) = "";
        sheet.Rating_1_to_5(k, 1) = NaN;
    end
    keyFile = fullfile(outDir, 'answer_key.csv');
    writetable(key, keyFile);
    writetable(sheet, fullfile(outDir, 'rating_sheet.csv'));
    fprintf('Wrote %d blinded clips + answer key + rating sheet to %s\n', ...
        size(clips, 1), outDir);
end
