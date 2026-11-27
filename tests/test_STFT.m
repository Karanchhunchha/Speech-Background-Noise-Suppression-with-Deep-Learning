classdef test_STFT < matlab.unittest.TestCase
    %TEST_STFT STFT/ISTFT round-trip and FFT resampling.

    methods (TestClassSetup)
        function addPaths(~)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(rootDir);
            addpath(genpath(fullfile(rootDir, 'src')));
        end
    end

    methods (Test)
        function testRoundTripHop256(testCase)
            rng(7);
            x = randn(16000, 1);
            S = stft(x, 512, 256);
            y = istft(S, 512, 256, numel(x));
            testCase.verifyEqual(numel(y), numel(x));
            testCase.verifyLessThan(max(abs(y - x)), 1e-10);
        end

        function testRoundTripHop128(testCase)
            rng(8);
            x = randn(12345, 1);
            S = stft(x, 512, 128);
            y = istft(S, 512, 128, numel(x));
            testCase.verifyLessThan(max(abs(y - x)), 1e-10);
        end

        function testResampleIdentity(testCase)
            x = randn(1000, 1);
            y = resampleFFT(x, 16000, 16000);
            testCase.verifyEqual(y, x);
        end

        function testResampleDownPreservesTone(testCase)
            fsIn = 48000;
            t = (0:fsIn - 1)' / fsIn;
            x = sin(2 * pi * 440 * t);
            y = resampleFFT(x, fsIn, 16000);
            testCase.verifyEqual(numel(y), 16000);
            % The 440 Hz tone survives: strong correlation with a 440 Hz
            % reference at the new rate.
            t2 = (0:16000 - 1)' / 16000;
            ref = sin(2 * pi * 440 * t2);
            c = (y' * ref) / (norm(y) * norm(ref));
            testCase.verifyGreaterThan(c, 0.99);
        end

        function testResampleUpLength(testCase)
            x = randn(1600, 1);
            y = resampleFFT(x, 16000, 48000);
            testCase.verifyEqual(numel(y), 4800);
        end
    end
end
