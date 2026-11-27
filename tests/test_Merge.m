classdef test_Merge < matlab.unittest.TestCase
    %TEST_MERGE Chunked-evaluation merge: combination and safety.

    methods (TestClassSetup)
        function addPaths(~)
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(rootDir);
            addpath(genpath(fullfile(rootDir, 'src')));
        end
    end

    methods (Test)
        function testMergeCombinesChunks(testCase)
            % Two disjoint chunk CSVs with known values must merge
            % into the exact combined means.
            dirPath = testCase.makeChunkDir();
            part1 = table(["f1"; "f2"], [1; 3], [10; 12], [0.5; 0.7], ...
                [2; 4], [9; 11], [0.6; 0.8], 'VariableNames', ...
                {'File', 'SegSNR_Noisy', 'LSD_Noisy', 'STOI_Noisy', ...
                 'SegSNR_CNN', 'LSD_CNN', 'STOI_CNN'});
            part2 = table("f3", 5, 14, 0.9, 6, 13, 1.0, 'VariableNames', ...
                {'File', 'SegSNR_Noisy', 'LSD_Noisy', 'STOI_Noisy', ...
                 'SegSNR_CNN', 'LSD_CNN', 'STOI_CNN'});
            writetable(part1, fullfile(dirPath, 'metrics_part1.csv'));
            writetable(part2, fullfile(dirPath, 'metrics_part2.csv'));

            summary = merge_evaluation(dirPath);
            testCase.verifyEqual(summary.SegSNR_dB(1), 3, 'AbsTol', 1e-12);
            testCase.verifyEqual(summary.SegSNR_dB(2), 4, 'AbsTol', 1e-12);
            testCase.verifyEqual(summary.STOI(1), 0.7, 'AbsTol', 1e-12);
            perFile = readtable(fullfile(dirPath, 'metrics_per_file.csv'));
            testCase.verifyEqual(height(perFile), 3);
            testCase.verifyTrue(exist(fullfile(dirPath, 'EVALUATION.md'), 'file') == 2);
        end

        function testMergeRejectsOverlappingChunks(testCase)
            % The same file in two chunks means two different runs are
            % being mixed — the merge must refuse, not average them.
            dirPath = testCase.makeChunkDir();
            part = table("f1", 1, 10, 0.5, 2, 9, 0.6, 'VariableNames', ...
                {'File', 'SegSNR_Noisy', 'LSD_Noisy', 'STOI_Noisy', ...
                 'SegSNR_CNN', 'LSD_CNN', 'STOI_CNN'});
            writetable(part, fullfile(dirPath, 'metrics_part1.csv'));
            writetable(part, fullfile(dirPath, 'metrics_part2.csv'));
            testCase.verifyError(@() merge_evaluation(dirPath), ...
                'merge_evaluation:OverlappingChunks');
        end
    end

    methods (Access = private)
        function dirPath = makeChunkDir(testCase)
            dirPath = tempname;
            mkdir(dirPath);
            testCase.addTeardown(@() rmdir(dirPath, 's'));
        end
    end
end
