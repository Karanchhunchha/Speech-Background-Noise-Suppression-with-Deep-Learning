classdef test_Network < matlab.unittest.TestCase
    %TEST_NETWORK Mask target, network construction and inference path.

    methods (TestClassSetup)
        function addPaths(~)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(rootDir);
            addpath(genpath(fullfile(rootDir, 'src')));
        end
    end

    methods (Test)
        function testIrmCleanOnlyIsOne(testCase)
            mag = rand(257, 40);
            mask = computeIRM(mag, zeros(257, 40));
            testCase.verifyEqual(mask, ones(257, 40), 'AbsTol', 1e-12);
        end

        function testIrmNoiseOnlyIsZero(testCase)
            mag = rand(257, 40);
            mask = computeIRM(zeros(257, 40), mag);
            testCase.verifyEqual(mask, zeros(257, 40), 'AbsTol', 1e-12);
        end

        function testIrmEqualParts(testCase)
            mag = rand(257, 40) + 0.1;
            mask = computeIRM(mag, mag);
            testCase.verifyEqual(mask, repmat(sqrt(0.5), 257, 40), 'AbsTol', 1e-6);
        end

        function testNetworkBuilds(testCase)
            layers = buildDenoiseNet(257, 64);
            testCase.verifyEqual(numel(layers), 15);
            testCase.verifyClass(layers(1), 'nnet.cnn.layer.ImageInputLayer');
            testCase.verifyClass(layers(end), 'nnet.cnn.layer.RegressionOutputLayer');
        end

        function testDenoiseWithAssembledNet(testCase)
            % An untrained (assembled) network must still run end to end:
            % correct output length and finite values. This exercises the
            % full inference path without a training run.
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            [noisy, fs] = audioread(pairs(1).noisyFile);
            layers = buildDenoiseNet(257, 64);
            model = struct('net', assembleNetwork(layers), 'mu', 0, 'sigma', 1, ...
                'winLen', 512, 'hop', 256, 'segLen', 64);
            out = denoiseWithNet(noisy, fs, model);
            testCase.verifyEqual(numel(out), numel(noisy));
            testCase.verifyTrue(all(isfinite(out)));
        end
    end
end
