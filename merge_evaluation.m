function summary = merge_evaluation()
%MERGE_EVALUATION Combine chunked evaluation outputs into final reports.
%   summary = MERGE_EVALUATION() reads every reports/metrics_part*.csv
%   file (the per-file outputs of chunked EVALUATE_MODELS runs, renamed
%   between chunks so later chunks do not overwrite earlier ones),
%   concatenates the per-file rows, and writes the final reports in
%   exactly the format EVALUATE_MODELS produces:
%   reports/metrics_per_file.csv, reports/metrics_summary.csv and
%   reports/EVALUATION.md. Chunked evaluation exists so the full
%   824-file test set can be evaluated across several sessions without
%   losing partial results.

    rootDir = setup();
    reportDir = fullfile(rootDir, 'reports');
    listing = dir(fullfile(reportDir, 'metrics_part*.csv'));
    names = {listing.name};
    names = names(~contains(names, 'summary'));
    if isempty(names)
        error('merge_evaluation:NoParts', ...
            'No reports/metrics_part*.csv files found in %s.', reportDir);
    end
    parts = cell(1, numel(names));
    for i = 1:numel(names)
        parts{i} = readtable(fullfile(reportDir, names{i}));
    end
    allRows = vertcat(parts{:});
    writetable(allRows, fullfile(reportDir, 'metrics_per_file.csv'));

    methodNames = {'Noisy', 'SpectralSubtraction', 'Wiener', 'CNN'};
    present = false(1, numel(methodNames));
    for m = 1:numel(methodNames)
        present(m) = any(strcmp(allRows.Properties.VariableNames, ...
            ['SegSNR_' methodNames{m}]));
    end
    methodNames = methodNames(present);
    numMethods = numel(methodNames);

    summary = table();
    summary.Method = string(methodNames)';
    segMean = zeros(numMethods, 1);
    segStd = zeros(numMethods, 1);
    lsdMean = zeros(numMethods, 1);
    stoiMean = zeros(numMethods, 1);
    stoiStd = zeros(numMethods, 1);
    for m = 1:numMethods
        segCol = allRows.(['SegSNR_' methodNames{m}]);
        lsdCol = allRows.(['LSD_' methodNames{m}]);
        stoiCol = allRows.(['STOI_' methodNames{m}]);
        segMean(m) = mean(segCol);
        segStd(m) = std(segCol);
        lsdMean(m) = mean(lsdCol);
        stoiMean(m) = mean(stoiCol);
        stoiStd(m) = std(stoiCol);
    end
    summary.SegSNR_dB = segMean;
    summary.SegSNR_Std = segStd;
    summary.LSD_dB = lsdMean;
    summary.STOI = stoiMean;
    summary.STOI_Std = stoiStd;
    writetable(summary, fullfile(reportDir, 'metrics_summary.csv'));

    fid = fopen(fullfile(reportDir, 'EVALUATION.md'), 'w');
    fprintf(fid, '# Objective evaluation\n\n');
    fprintf(fid, 'Data: VoiceBank-DEMAND standard test set (data/voicebank). ');
    fprintf(fid, 'Files evaluated: %d (combined from %d chunked runs). ', height(allRows), numel(names));
    fprintf(fid, 'Metrics are means over files (segmental SNR and STOI: higher is better; LSD: lower is better).\n\n');
    fprintf(fid, '| Method | Segmental SNR (dB) | LSD (dB) | STOI |\n');
    fprintf(fid, '|---|---|---|---|\n');
    for m = 1:numMethods
        fprintf(fid, '| %s | %.2f | %.2f | %.3f |\n', methodNames{m}, ...
            segMean(m), lsdMean(m), stoiMean(m));
    end
    cnnIdx = find(strcmp(methodNames, 'CNN'), 1);
    if ~isempty(cnnIdx)
        fprintf(fid, '\nSegmental-SNR improvement of the CNN over the noisy input: %.2f dB.\n', ...
            segMean(cnnIdx) - segMean(1));
    end
    fclose(fid);
    fprintf('Merged %d chunk files into final reports (%d files total).\n', ...
        numel(names), height(allRows));
    disp(summary);
end
