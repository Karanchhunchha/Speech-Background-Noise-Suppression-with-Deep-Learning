classdef test_ToolboxCrossCheck < matlab.unittest.TestCase
%TEST_TOOLBOXCROSSCHECK Hand-written DSP vs the Signal Processing Toolbox.
%   The project's STFT/ISTFT are hand-written so the pipeline needs no
%   toolbox beyond Deep Learning. When the Signal Processing Toolbox
%   is present, these tests cross-check the hand-written STFT against
%   the toolbox's own stft() under matched settings (periodic Hann
%   window, same length/hop, centred, one-sided). When the toolbox is
%   absent the tests are marked Incomplete — never silently passed.

    methods (Test)
        function stftMatchesToolbox(testCase)
            toolboxStft = findToolboxStft(testCase);
            rng(7);
            x = randn(16000, 1);
            ours = stft(x, 512, 256);
            % Note: the toolbox parser rejects 'Centered' together
            % with 'FrequencyRange', so the range is left at its
            % default and a two-sided result is sliced to one-sided.
            theirs = toolboxStft(x, 'Window', hann(512, 'periodic'), ...
                'OverlapLength', 256, 'FFTLength', 512, ...
                'Centered', true);
            if size(theirs, 1) == 512
                theirs = theirs(1:257, :);
            end
            nFrames = min(size(ours, 2), size(theirs, 2));
            testCase.verifyEqual(size(theirs, 1), size(ours, 1));
            diff = max(abs(ours(:, 1:nFrames) - theirs(:, 1:nFrames)), [], 'all');
            scale = max(abs(ours), [], 'all');
            testCase.verifyLessThan(diff / scale, 1e-6, ...
                'Hand-written STFT disagrees with the Signal Processing Toolbox stft.');
        end
    end
end

function fh = findToolboxStft(testCase)
%FINDTOOLBOXSTFT Handle to the toolbox stft, or an Incomplete verdict.
    candidates = which('stft', '-all');
    if ischar(candidates)
        candidates = {candidates};
    end
    fh = [];
    for i = 1:numel(candidates)
        if contains(candidates{i}, 'toolbox') && contains(candidates{i}, 'signal')
            % Invoke the toolbox implementation via its own folder
            % (the current folder outranks the path, so the toolbox
            % stft shadows this project's hand-written one there).
            fh = @(varargin) toolboxStftImpl(candidates{i}, varargin{:});
            return;
        end
    end
    testCase.assumeTrue(false, ...
        'Signal Processing Toolbox stft not found; cross-check cannot run here.');
end

function S = toolboxStftImpl(file, varargin)
%TOOLBOXSTFTIMPL Invoke the toolbox stft file shadowing our own stft.
    currentDir = pwd;
    cleanup = onCleanup(@() cd(currentDir));
    cd(fileparts(file));
    S = stft(varargin{:});
    clear cleanup;
end
