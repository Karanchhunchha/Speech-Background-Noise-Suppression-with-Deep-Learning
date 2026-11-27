function ok = run_all_tests()
%RUN_ALL_TESTS Run the project's full unit-test suite.
%   ok = RUN_ALL_TESTS() runs every test under tests/ and prints the
%   pass/fail/incomplete summary. Returns true when nothing failed.
%   The suite needs only the committed sample data (data/sample), so it
%   runs on a fresh clone with no downloads.

    rootDir = setup();
    results = runtests(fullfile(rootDir, 'tests'), 'IncludeSubfolders', true);
    numPassed = sum([results.Passed]);
    numFailed = sum([results.Failed]);
    numIncomplete = sum([results.Incomplete]);
    fprintf('\n--- TEST SUITE SUMMARY ---\n');
    fprintf('Passed : %d\n', numPassed);
    fprintf('Failed : %d\n', numFailed);
    fprintf('Incomplete : %d\n', numIncomplete);
    if numFailed > 0
        failedNames = {results([results.Failed]).Name};
        fprintf('Failed tests:\n');
        for i = 1:numel(failedNames)
            fprintf('  - %s\n', failedNames{i});
        end
    end
    ok = (numFailed == 0);
end
