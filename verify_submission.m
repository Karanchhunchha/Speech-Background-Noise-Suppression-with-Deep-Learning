function ok = verify_submission()
%VERIFY_SUBMISSION One-command reviewer self-check for this repository.
%   ok = VERIFY_SUBMISSION() runs, in order:
%     1. environment report (MATLAB release, products installed vs the
%        project's required-products list),
%     2. the full unit/integration test suite (RUN_ALL_TESTS),
%     3. a quick re-evaluation of the first 8 test files, compared
%        row-by-row against the committed reports/metrics_per_file.csv
%        (tolerances: 0.05 dB SegSNR/LSD/SI-SDR, 0.005 STOI) — proves
%        the shipped model + code reproduce the published numbers,
%     4. README results-table consistency with reports/
%        metrics_summary.csv,
%     5. a scan of the project's .m/.md files for absolute paths,
%     6. a load check of the shipped model artifact.
%   Every check prints PASS/FAIL/SKIP with its evidence and the
%   function returns true only if nothing FAILED. Designed to finish
%   in a few minutes on a free cloud session.

    rootDir = setup();
    failures = 0;
    fprintf('=== verify_submission: %s ===\n', rootDir);
    fprintf('MATLAB %s (%s)\n', version, version('-release'));

    % 1. Required products -------------------------------------------
    reqFile = fullfile(rootDir, 'reports', 'required_products.txt');
    if exist(reqFile, 'file')
        required = strtrim(string(splitlines(fileread(reqFile))));
        required = required(required ~= "");
        installed = string({ver().Name})';
        for i = 1:numel(required)
            if any(installed == required(i))
                fprintf('PASS  product present: %s\n', required(i));
            else
                fprintf('FAIL  product missing: %s\n', required(i));
                failures = failures + 1;
            end
        end
    else
        fprintf('SKIP  reports/required_products.txt not present.\n');
    end

    % 2. Test suite ----------------------------------------------------
    if run_all_tests()
        fprintf('PASS  test suite\n');
    else
        fprintf('FAIL  test suite\n');
        failures = failures + 1;
    end

    % 3. Reproduce committed metrics on 8 files ------------------------
    perFileCsv = fullfile(rootDir, 'reports', 'metrics_per_file.csv');
    if exist(perFileCsv, 'file') && ...
            exist(fullfile(rootDir, 'data', 'voicebank', 'clean_test'), 'dir')
        committed = readtable(perFileCsv);
        before = snapshotReports(rootDir);
        cleanup = onCleanup(@() restoreReports(rootDir, before));
        evaluate_models('MaxFiles', 8);
        fresh = readtable(perFileCsv);
        methodNames = {'Noisy', 'SpectralSubtraction', 'Wiener', 'CNN'};
        worst = 0;
        for m = 1:numel(methodNames)
            for metric = {'SegSNR_', 'LSD_', 'STOI_', 'SISDR_'}
                col = [metric{1} methodNames{m}];
                if any(strcmp(fresh.Properties.VariableNames, col)) && ...
                        any(strcmp(committed.Properties.VariableNames, col))
                    d = max(abs(fresh.(col)(1:8) - committed.(col)(1:8)), [], 'omitnan');
                    if startsWith(metric{1}, 'STOI')
                        worst = max(worst, d / 0.005);
                    else
                        worst = max(worst, d / 0.05);
                    end
                end
            end
        end
        if worst <= 1
            fprintf('PASS  committed metrics reproduced on 8 files (worst %.2f of tolerance)\n', worst);
        else
            fprintf('FAIL  committed metrics NOT reproduced (worst %.2f of tolerance)\n', worst);
            failures = failures + 1;
        end
        clear cleanup;
    else
        fprintf('SKIP  metric reproduction (needs the dataset + committed reports).\n');
    end

    % 4. README table consistency --------------------------------------
    readmeFile = fullfile(rootDir, 'README.md');
    if exist(readmeFile, 'file') && contains(fileread(readmeFile), 'RESULTS:BEGIN')
        if update_readme_results('check')
            fprintf('PASS  README results table matches metrics_summary.csv\n');
        else
            fprintf('FAIL  README results table is stale (run update_readme_results()).\n');
            failures = failures + 1;
        end
    else
        fprintf('SKIP  README results markers not found.\n');
    end

    % 5. Absolute-path scan --------------------------------------------
    % (patterns are concatenated so this file does not match itself)
    badPatterns = {['/MATLAB' ' Drive'], ['/home' '/'], ...
                   ['C:' '\Users'], ['/Users' '/']};
    offenders = {};
    files = [dir(fullfile(rootDir, '**', '*.m')); dir(fullfile(rootDir, '**', '*.md'))];
    for i = 1:numel(files)
        f = fullfile(files(i).folder, files(i).name);
        if strcmp(files(i).name, 'verify_submission.m')
            continue;
        end
        text = fileread(f);
        for j = 1:numel(badPatterns)
            if contains(text, badPatterns{j})
                offenders{end + 1} = f; %#ok<AGROW>
                break;
            end
        end
    end
    if isempty(offenders)
        fprintf('PASS  no absolute paths in .m/.md files\n');
    else
        fprintf('FAIL  absolute paths found in: %s\n', strjoin(offenders, ', '));
        failures = failures + 1;
    end

    % 6. Model artifact -------------------------------------------------
    try
        model = loadDenoiseModel(fullfile(rootDir, 'models', 'denoiseNet.mat'));
        if isfield(model, 'net') && isfield(model, 'mu')
            fprintf('PASS  shipped model loads (denoiseNet.mat)\n');
        else
            fprintf('FAIL  shipped model is missing fields.\n');
            failures = failures + 1;
        end
    catch ME
        fprintf('FAIL  shipped model does not load: %s\n', ME.message);
        failures = failures + 1;
    end

    ok = failures == 0;
    if ok
        fprintf('=== verify_submission: ALL CHECKS PASSED ===\n');
    else
        fprintf('=== verify_submission: %d CHECK(S) FAILED ===\n', failures);
    end
end

function snap = snapshotReports(rootDir)
%SNAPSHOTREPORTS Read the committed report files into memory.
    names = {'metrics_per_file.csv', 'metrics_summary.csv', ...
             'metrics_by_snr.csv', 'EVALUATION.md'};
    snap = cell(0, 2);
    for i = 1:numel(names)
        f = fullfile(rootDir, 'reports', names{i});
        if exist(f, 'file')
            snap(end + 1, :) = {names{i}, fileread(f)}; %#ok<AGROW>
        end
    end
end

function restoreReports(rootDir, snap)
%RESTOREREPORTS Put the committed report files back after the quick eval.
    for i = 1:size(snap, 1)
        fid = fopen(fullfile(rootDir, 'reports', snap{i, 1}), 'w');
        fwrite(fid, snap{i, 2});
        fclose(fid);
    end
end
