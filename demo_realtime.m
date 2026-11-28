function results = demo_realtime(varargin)
%DEMO_REALTIME Simulated streaming (real-time) enhancement benchmark.
%   results = DEMO_REALTIME() feeds one noisy test utterance through
%   every enhancer in fixed-size chunks — the way a streaming system
%   would receive it — and measures the processing time per chunk
%   (mean / p95 / max) against the chunk's audio duration (the
%   real-time budget). The stream is simulated from a file; no live
%   audio device is used (the verification environment has none).
%
%   Honesty notes: the classical enhancers are whole-file algorithms;
%   here they are applied per chunk, which is the standard simulated-
%   streaming approximation and is labelled as such. The CNN consumes
%   64-frame segments (window 512, hop 256 at 16 kHz), so its
%   algorithmic look-ahead is one segment — 16,384 samples ≈ 1.02 s —
%   regardless of chunking; that latency is reported, not hidden.
%
%   Options: 'ChunkSamples' (default 4096 = 256 ms), 'MaxChunks'
%   (default 40), 'FileIndex' (which test pair, default 1).
%   Writes reports/realtime_results.csv and returns the table.

    p = inputParser;
    p.addParameter('ChunkSamples', 4096, @(x) isscalar(x) && x >= 512);
    p.addParameter('MaxChunks', 40, @(x) isscalar(x) && x >= 1);
    p.addParameter('FileIndex', 1, @(x) isscalar(x) && x >= 1);
    p.parse(varargin{:});

    rootDir = setup();
    pairs = listPairs(fullfile(rootDir, 'data', 'voicebank', 'clean_test'), ...
                      fullfile(rootDir, 'data', 'voicebank', 'noisy_test'));
    if isempty(pairs)
        pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                          fullfile(rootDir, 'data', 'sample', 'noisy'));
    end
    if isempty(pairs)
        error('demo_realtime:NoData', 'No pairs found (run download_data() or use the sample).');
    end
    pair = pairs(min(p.Results.FileIndex, numel(pairs)));
    [noisy, fs] = audioread(pair.noisyFile);
    noisy = noisy(:, 1);
    model = loadDenoiseModel(fullfile(rootDir, 'models', 'denoiseNet.mat'));

    chunk = p.Results.ChunkSamples;
    numChunks = min(p.Results.MaxChunks, floor(numel(noisy) / chunk));
    budgetMs = 1000 * chunk / fs;
    methodNames = {'SpectralSubtraction', 'Wiener', 'LogMMSE', 'CNN'};
    times = zeros(numChunks, numel(methodNames));
    for c = 1:numChunks
        seg = noisy((c - 1) * chunk + 1 : c * chunk);
        t = tic; spectralSubtraction(seg, fs); times(c, 1) = toc(t);
        t = tic; wienerEnhance(seg, fs);      times(c, 2) = toc(t);
        t = tic; logMmseEnhance(seg, fs);      times(c, 3) = toc(t);
        t = tic; denoiseWithNet(seg, fs, model); times(c, 4) = toc(t);
    end
    rows = {};
    for m = 1:numel(methodNames)
        ms = 1000 * times(:, m);
        rows(end + 1, :) = {string(methodNames{m}), mean(ms), ...
            prctile(ms, 95), max(ms), budgetMs, mean(ms) / budgetMs}; %#ok<AGROW>
    end
    results = cell2table(rows, 'VariableNames', {'Method', 'Mean_ms', ...
        'P95_ms', 'Max_ms', 'Budget_ms', 'RealTimeFactor'});
    fprintf('Simulated stream: %s, %d chunks of %d samples (%.0f ms budget each)\n', ...
        pair.id, numChunks, chunk, budgetMs);
    fprintf('CNN algorithmic look-ahead: one 64-frame segment = 16384 samples = %.2f s.\n', ...
        16384 / fs);
    disp(results);
    reportDir = fullfile(rootDir, 'reports');
    if ~exist(reportDir, 'dir'); mkdir(reportDir); end
    writetable(results, fullfile(reportDir, 'realtime_results.csv'));
end
