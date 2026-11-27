function demo_denoise(varargin)
%DEMO_DENOISE Before/after demo on the committed sample pairs.
%   DEMO_DENOISE() runs every enhancer on the first sample pairs in
%   data/sample and writes, per file, the clean reference, the noisy
%   input and each enhanced version as .wav files under results/demo,
%   plus a spectrogram comparison figure (PNG). Listening to the
%   noisy/CNN pair is the subjective evaluation the project brief asks
%   for alongside the objective metrics.
%
%   Name-value options:
%     'NumFiles'  sample files to process (default 2)

    p = inputParser;
    addParameter(p, 'NumFiles', 2);
    parse(p, varargin{:});

    rootDir = setup();
    pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                      fullfile(rootDir, 'data', 'sample', 'noisy'));
    if isempty(pairs)
        error('demo_denoise:NoSample', 'No sample pairs found under data/sample.');
    end
    pairs = pairs(1:min(numel(pairs), p.Results.NumFiles));

    modelFile = fullfile(rootDir, 'models', 'denoiseNet.mat');
    haveModel = exist(modelFile, 'file') == 2;
    if haveModel
        model = loadDenoiseModel(modelFile);
    else
        fprintf(['No trained model found; the demo will contain the ' ...
            'classical methods only. Run train_model() first for the CNN output.\n']);
    end

    outDir = fullfile(rootDir, 'results', 'demo');
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    for i = 1:numel(pairs)
        [clean, fs] = audioread(pairs(i).cleanFile);
        [noisy, fsNoisy] = audioread(pairs(i).noisyFile);
        if size(clean, 2) > 1 || size(noisy, 2) > 1
            error('demo_denoise:StereoNotSupported', ...
                'Pair %s has more than one channel; this project expects mono audio.', pairs(i).id);
        end
        if fsNoisy ~= fs
            error('demo_denoise:RateMismatch', ...
                'Pair %s: clean is %d Hz but noisy is %d Hz.', pairs(i).id, fs, fsNoisy);
        end
        n = min(numel(clean), numel(noisy));
        clean = clean(1:n);
        noisy = noisy(1:n);
        spec = spectralSubtraction(noisy, fs);
        wien = wienerEnhance(noisy, fs);
        audiowrite(fullfile(outDir, pairs(i).id + "_clean.wav"), clean, fs);
        audiowrite(fullfile(outDir, pairs(i).id + "_noisy.wav"), noisy, fs);
        audiowrite(fullfile(outDir, pairs(i).id + "_specsub.wav"), spec, fs);
        audiowrite(fullfile(outDir, pairs(i).id + "_wiener.wav"), wien, fs);
        panels = {noisy, spec, wien};
        titles = {'Noisy', 'Spectral subtraction', 'Wiener'};
        if haveModel
            cnn = denoiseWithNet(noisy, fs, model);
            audiowrite(fullfile(outDir, pairs(i).id + "_cnn.wav"), cnn, fs);
            panels{end + 1} = cnn;
            titles{end + 1} = 'CNN (this project)';
        end
        fig = figure('Visible', 'off', 'Position', [100 100 900 700]);
        figCleanup = onCleanup(@() close(fig));
        for k = 1:numel(panels)
            subplot(numel(panels), 1, k);
            S = stft(panels{k}, 512, 256);
            imagesc(20 * log10(abs(S) + 1e-6));
            axis xy;
            colorbar;
            ylabel('Freq bin');
            title(titles{k});
        end
        xlabel('Frame');
        saveas(fig, fullfile(outDir, pairs(i).id + "_spectrograms.png"));
        clear figCleanup;
        fprintf('Demo outputs for %s written to %s\n', pairs(i).id, outDir);
    end
end
