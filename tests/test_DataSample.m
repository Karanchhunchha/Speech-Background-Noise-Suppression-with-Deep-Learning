classdef test_DataSample < matlab.unittest.TestCase
    %TEST_DATASAMPLE The committed sample data is complete and consistent.

    methods (TestClassSetup)
        function addPaths(~)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(rootDir);
            addpath(genpath(fullfile(rootDir, 'src')));
        end
    end

    methods (Test)
        function testEightPairsPresent(testCase)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            testCase.verifyEqual(numel(pairs), 8);
        end

        function testPairsAre16kAndMatched(testCase)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            pairs = listPairs(fullfile(rootDir, 'data', 'sample', 'clean'), ...
                              fullfile(rootDir, 'data', 'sample', 'noisy'));
            for i = 1:numel(pairs)
                [clean, fsC] = audioread(pairs(i).cleanFile);
                [noisy, fsN] = audioread(pairs(i).noisyFile);
                testCase.verifyEqual(fsC, 16000);
                testCase.verifyEqual(fsN, 16000);
                testCase.verifyEqual(numel(clean), numel(noisy));
                % Noisy must actually differ from clean (it contains
                % added DEMAND noise).
                testCase.verifyGreaterThan(max(abs(clean - noisy)), 0.01);
            end
        end
    end
end
