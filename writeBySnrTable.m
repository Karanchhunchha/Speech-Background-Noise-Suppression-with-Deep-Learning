function bySnr = writeBySnrTable(reportDir, perFile, methodNames)
%WRITEBYSNRTABLE Per-SNR breakdown table from per-file evaluation rows.
%   bySnr = WRITEBYSNRTABLE(reportDir, perFile, methodNames) groups the
%   rows of a metrics_per_file-style table by their measured input SNR
%   (nearest nominal test level, see SNRBUCKET) and writes
%   reports/metrics_by_snr.csv with per-level, per-method means.
%   Returns the table. Rows whose InputSNR_dB is not finite are
%   skipped.

    levels = [2.5, 7.5, 12.5, 17.5];
    snr = perFile.InputSNR_dB;
    bucket = snrBucket(snr);
    rows = {};
    for L = levels
        sel = isfinite(snr) & bucket == L;
        if ~any(sel)
            continue;
        end
        for m = 1:numel(methodNames)
            name = methodNames{m};
            sisdrMean = NaN;
            if any(strcmp(perFile.Properties.VariableNames, ['SISDR_' name]))
                sisdrMean = mean(perFile.(['SISDR_' name])(sel), 'omitnan');
            end
            rows(end + 1, :) = {L, string(name), sum(sel), ...
                mean(perFile.(['SegSNR_' name])(sel), 'omitnan'), ...
                mean(perFile.(['LSD_' name])(sel), 'omitnan'), ...
                mean(perFile.(['STOI_' name])(sel), 'omitnan'), ...
                sisdrMean}; %#ok<AGROW>
        end
    end
    bySnr = cell2table(rows, 'VariableNames', ...
        {'SNR_dB', 'Method', 'N', 'SegSNR_dB', 'LSD_dB', 'STOI', 'SISDR_dB'});
    writetable(bySnr, fullfile(reportDir, 'metrics_by_snr.csv'));
end
