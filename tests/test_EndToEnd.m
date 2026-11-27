classdef test_EndToEnd < matlab.unittest.TestCase
    %TEST_ENDTOEND Full enhancement + metric chain on a real sample pair.

    methods (TestClassSetup)
        function addPaths(~)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(rootDir);
            addpath(genpath(fullfile(rootDir, 'src')));
        end
    end

    methods (Test)
        function testFullChainOnSample(testCase)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            [clean, fs] = audioread(pairs(1).cleanFile);
            [noisy, ~] = audioread(pairs(1).noisyFile);
            n = min(numel(clean), numel(noisy));
            clean = clean(1:n);
            noisy = noisy(1:n);

            enhancers = {@spectralSubtraction, @wienerEnhance};
            for k = 1:numel(enhancers)
                out = enhancers{k}(noisy, fs);
                testCase.verifyEqual(numel(out), n);
                testCase.verifyTrue(all(isfinite(out)));
                s = segSnrDb(clean, out);
                l = lsdDb(clean, out);
                st = stoiScore(clean, out, fs);
                testCase.verifyTrue(isfinite(s));
                testCase.verifyTrue(isfinite(l));
                testCase.verifyGreaterThanOrEqual(st, 0);
                testCase.verifyLessThanOrEqual(st, 1);
            end
            % Both classical methods must not make the sample worse in
            % segmental SNR than the noisy input.
            base = segSnrDb(clean, noisy);
            testCase.verifyGreaterThan(segSnrDb(clean, spectralSubtraction(noisy, fs)), base - 1);
            testCase.verifyGreaterThan(segSnrDb(clean, wienerEnhance(noisy, fs)), base - 1);
        end
    end
end
