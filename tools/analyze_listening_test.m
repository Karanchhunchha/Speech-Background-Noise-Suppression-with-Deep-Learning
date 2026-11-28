function results = analyze_listening_test(varargin)
%ANALYZE_LISTENING_TEST Summarise collected listening-test ratings.
%   results = ANALYZE_LISTENING_TEST() reads the filled rating sheet
%   and the answer key in 'TestDir' (default 'listening_test', as
%   produced by MAKE_LISTENING_TEST) and reports, per method, the mean
%   rating and an approximate 95% confidence interval (normal
%   approximation, 1.96 * SE — no Statistics Toolbox required).
%
%   If no ratings have been collected yet, it says so plainly and
%   returns an empty table — ratings are never invented by this tool.

    p = inputParser;
    p.addParameter('TestDir', 'listening_test', @ischar);
    p.parse(varargin{:});
    rootDir = setup();
    testDir = fullfile(rootDir, p.Results.TestDir);
    sheetFile = fullfile(testDir, 'rating_sheet.csv');
    keyFile = fullfile(testDir, 'answer_key.csv');
    if ~exist(sheetFile, 'file') || ~exist(keyFile, 'file')
        error('analyze_listening_test:MissingFiles', ...
            'Run make_listening_test() first (missing sheet or key in %s).', testDir);
    end
    sheet = readtable(sheetFile);
    key = readtable(keyFile);
    rated = sheet(~isnan(sheet.Rating_1_to_5), :);
    if isempty(rated)
        fprintf('No ratings collected yet — the results table is empty by design.\n');
        results = table();
        return;
    end
    joined = innerjoin(rated, key, 'Keys', 'Clip');
    methods = unique(joined.Method, 'stable');
    rows = {};
    for m = 1:numel(methods)
        r = joined.Rating_1_to_5(joined.Method == methods(m));
        n = numel(r);
        mu = mean(r);
        if n > 1
            half = 1.96 * std(r) / sqrt(n);
        else
            half = NaN;
        end
        rows(end + 1, :) = {methods(m), n, mu, mu - half, mu + half}; %#ok<AGROW>
    end
    results = cell2table(rows, 'VariableNames', ...
        {'Method', 'NumRatings', 'MeanRating', 'CI95_Low', 'CI95_High'});
    disp(results);
end
