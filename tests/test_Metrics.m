classdef test_Metrics < matlab.unittest.TestCase
    %TEST_METRICS Sanity properties of the objective metrics.

    methods (TestClassSetup)
        function addPaths(~)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(rootDir);
            addpath(genpath(fullfile(rootDir, 'src')));
        end
    end

    methods (Test)
        function testSnrPerfectIsHuge(testCase)
            % Error power is floored at eps, so a perfect estimate
            % returns a very large finite SNR (not Inf).
            x = randn(8000, 1);
            testCase.verifyGreaterThan(snrDb(x, x), 100);
        end

        function testSegSnrPerfectIsClipMax(testCase)
            rng(3);
            x = randn(16000, 1);
            testCase.verifyEqual(segSnrDb(x, x), 35, 'AbsTol', 1e-9);
        end

        function testLsdPerfectIsZero(testCase)
            rng(4);
            x = randn(16000, 1);
            testCase.verifyEqual(lsdDb(x, x), 0, 'AbsTol', 1e-9);
        end

        function testStoiIdentity(testCase)
            % Real speech from the committed sample.
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            testCase.verifyNotEmpty(pairs);
            [x, fs] = audioread(pairs(1).cleanFile);
            testCase.verifyEqual(stoiScore(x, x, fs), 1, 'AbsTol', 1e-6);
        end

        function testStoiDecreasesWithNoise(testCase)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            [x, fs] = audioread(pairs(1).cleanFile);
            rng(11);
            noise = randn(size(x));
            noise = noise / norm(noise) * norm(x);          % 0 dB SNR mix
            mix0 = x + noise;
            mix10 = x + noise * 10 ^ (-10 / 20);             % 10 dB SNR mix
            s0 = stoiScore(x, mix0, fs);
            s10 = stoiScore(x, mix10, fs);
            testCase.verifyGreaterThan(s10, s0);
            testCase.verifyLessThan(s0, 1);
            testCase.verifyGreaterThan(s0, 0);
        end

        function testStoiOfActualNoisySample(testCase)
            % The committed noisy sample must score below the clean
            % reference but in the plausible VoiceBank range.
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            [x, fs] = audioread(pairs(1).cleanFile);
            [y, ~] = audioread(pairs(1).noisyFile);
            s = stoiScore(x, y, fs);
            testCase.verifyGreaterThan(s, 0.3);
            testCase.verifyLessThan(s, 1);
        end
    end
end
