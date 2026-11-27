classdef test_Baselines < matlab.unittest.TestCase
    %TEST_BASELINES Classical enhancers on a synthetic speech-like mixture.

    methods (TestClassSetup)
        function addPaths(~)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(rootDir);
            addpath(genpath(fullfile(rootDir, 'src')));
        end
    end

    methods (Static, Access = private)
        function [clean, noisy] = makeMixture()
            % Amplitude-gated harmonic "speech" (bursts with pauses, so
            % the signal is sparse in time-frequency like real speech)
            % plus white noise at about 5 dB SNR.
            fs = 16000;
            t = (0:fs * 3 - 1)' / fs;
            rng(21);
            f0 = 120;
            sig = zeros(size(t));
            for h = 1:6
                sig = sig + (1 / h) * sin(2 * pi * f0 * h * t + rand * 2 * pi);
            end
            gate = double(mod(t, 0.5) < 0.3);
            env = smoothdata(gate, 'movmean', 200);
            clean = sig .* env;
            clean = clean / max(abs(clean)) * 0.7;
            noise = randn(size(t));
            noise = noise / sqrt(mean(noise .^ 2)) * sqrt(mean(clean .^ 2)) * 10 ^ (-5 / 20);
            noisy = clean + noise;
        end
    end

    methods (Test)
        function testSpectralSubtractionImproves(testCase)
            [clean, noisy] = test_Baselines.makeMixture();
            out = spectralSubtraction(noisy, 16000);
            testCase.verifyEqual(numel(out), numel(noisy));
            testCase.verifyTrue(all(isfinite(out)));
            testCase.verifyGreaterThan(segSnrDb(clean, out), segSnrDb(clean, noisy));
        end

        function testWienerImproves(testCase)
            [clean, noisy] = test_Baselines.makeMixture();
            out = wienerEnhance(noisy, 16000);
            testCase.verifyEqual(numel(out), numel(noisy));
            testCase.verifyTrue(all(isfinite(out)));
            testCase.verifyGreaterThan(segSnrDb(clean, out), segSnrDb(clean, noisy));
        end

        function testBaselinesRejectBadFs(testCase)
            testCase.verifyError(@() spectralSubtraction(randn(100, 1), -1), ...
                'spectralSubtraction:BadFs');
            testCase.verifyError(@() wienerEnhance(randn(100, 1), 0), ...
                'wienerEnhance:BadFs');
        end
    end
end
