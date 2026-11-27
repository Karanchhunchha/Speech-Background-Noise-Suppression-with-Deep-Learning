function ok = update_readme_results(mode)
%UPDATE_README_RESULTS Keep the README results table generated, not typed.
%   update_readme_results() regenerates the markdown table between the
%   <!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> markers in
%   README.md from reports/metrics_summary.csv, so the README can
%   never silently drift from the measured numbers.
%   ok = update_readme_results('check') only verifies and returns
%   whether the table is already up to date (used by
%   VERIFY_SUBMISSION).

    if nargin < 1
        mode = 'write';
    end
    rootDir = setup();
    summary = readtable(fullfile(rootDir, 'reports', 'metrics_summary.csv'));
    lines = {'| Method | SegSNR (dB) | LSD (dB) | STOI | SI-SDR (dB) |', ...
             '|---|---|---|---|---|'};
    for i = 1:height(summary)
        sisdr = NaN;
        if any(strcmp(summary.Properties.VariableNames, 'SISDR_dB'))
            sisdr = summary.SISDR_dB(i);
        end
        lines{end + 1} = sprintf('| %s | %.2f | %.2f | %.3f | %.2f |', ...
            summary.Method{i}, summary.SegSNR_dB(i), summary.LSD_dB(i), ...
            summary.STOI(i), sisdr); %#ok<AGROW>
    end
    tableText = strjoin(lines, newline);

    readmeFile = fullfile(rootDir, 'README.md');
    text = fileread(readmeFile);
    beginMark = '<!-- RESULTS:BEGIN -->';
    endMark = '<!-- RESULTS:END -->';
    bi = strfind(text, beginMark);
    ei = strfind(text, endMark);
    if isempty(bi) || isempty(ei)
        error('update_readme_results:NoMarkers', ...
            'README.md has no RESULTS:BEGIN/END markers.');
    end
    if strcmp(mode, 'check')
        ok = contains(text, tableText);
        return;
    end
    newText = [text(1:bi + length(beginMark) - 1), newline, tableText, ...
        newline, text(ei:end)];
    fid = fopen(readmeFile, 'w');
    fwrite(fid, newText);
    fclose(fid);
    ok = true;
    fprintf('README results table regenerated from metrics_summary.csv\n');
end
