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
            % Magnitudes are kept away from zero: with an eps floor in
            % the IRM denominator, near-zero bins cannot reach exactly 1.
            mag = rand(257, 40) + 0.5;
            mask = computeIRM(mag, zeros(257, 40));
            testCase.verifyEqual(mask, ones(257, 40), 'AbsTol', 1e-9);
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

        function testV3GraphBuildsAndPredicts(testCase)
            net = dlnetwork(buildDenoiseNetV3(257, 64));
            out = predict(net, randn(257, 64, 3, 1, 'single'));
            testCase.verifyEqual(size(out), [257, 64, 2, 1]);
            testCase.verifyTrue(all(isfinite(out), 'all'));
        end

        function testCrmLossIsFiniteWithGradients(testCase)
            net = dlnetwork(buildDenoiseNetV3(257, 64));
            X = dlarray(randn(257, 64, 3, 2, 'single'), 'SSCB');
            T = dlarray(randn(257, 64, 2, 2, 'single') * 0.1, 'SSCB');
            [loss, gradients] = dlfeval(@modelLossCRM, net, X, T);
            testCase.verifyTrue(isfinite(double(extractdata(loss))));
            testCase.verifyEqual(height(gradients), height(net.Learnables));
        end

        function testDenoiseCrmSmoke(testCase)
            % An untrained complex-mask model must still run end to
            % end through DENOISEWITHNET's 'crm' branch: correct
            % output length and finite values.
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            [noisy, fs] = audioread(pairs(1).noisyFile);
            model = struct('net', dlnetwork(buildDenoiseNetV3(257, 64)), ...
                'mu', 0, 'sigma', 1, 'specScale', 1, 'maskBound', 2, ...
                'kind', 'crm', 'winLen', 512, 'hop', 256, 'segLen', 64);
            out = denoiseWithNet(noisy, fs, model);
            testCase.verifyEqual(numel(out), numel(noisy));
            testCase.verifyTrue(all(isfinite(out)));
        end

        function testDenoiseWithTinyTrainedNet(testCase)
            % A network trained for one epoch on random patches must run
            % end to end through DENOISEWITHNET: correct output length
            % and finite values. (assembleNetwork cannot be used here —
            % it refuses layers whose parameters are not yet trained.)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            [noisy, fs] = audioread(pairs(1).noisyFile);
            rng(5);
            X = randn(257, 64, 1, 16, 'single');
            Y = rand(257, 64, 1, 16, 'single');
            opts = trainingOptions('adam', 'MaxEpochs', 1, 'MiniBatchSize', 8, ...
                'Verbose', false, 'Plots', 'none', 'ExecutionEnvironment', 'cpu');
            net = trainNetwork(X, Y, buildDenoiseNet(257, 64), opts);
            model = struct('net', net, 'mu', 0, 'sigma', 1, ...
                'winLen', 512, 'hop', 256, 'segLen', 64);
            out = denoiseWithNet(noisy, fs, model);
            testCase.verifyEqual(numel(out), numel(noisy));
            testCase.verifyTrue(all(isfinite(out)));
        end

        function testDenoiseWithNetShortFile(testCase)
            % A file shorter than one 64-frame segment exercises the
            % zero-padding branch of DENOISEWITHNET: the output must
            % still match the input length and stay finite.
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            [noisy, fs] = audioread(pairs(1).noisyFile);
            noisy = noisy(1:2000);
            rng(5);
            X = randn(257, 64, 1, 16, 'single');
            Y = rand(257, 64, 1, 16, 'single');
            opts = trainingOptions('adam', 'MaxEpochs', 1, 'MiniBatchSize', 8, ...
                'Verbose', false, 'Plots', 'none', 'ExecutionEnvironment', 'cpu');
            net = trainNetwork(X, Y, buildDenoiseNet(257, 64), opts);
            model = struct('net', net, 'mu', 0, 'sigma', 1, ...
                'winLen', 512, 'hop', 256, 'segLen', 64);
            out = denoiseWithNet(noisy, fs, model);
            testCase.verifyEqual(numel(out), numel(noisy));
            testCase.verifyTrue(all(isfinite(out)));
        end
    end
end
