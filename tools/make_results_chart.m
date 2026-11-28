function make_results_chart()
%MAKE_RESULTS_CHART Bar chart of the headline results from the CSV.
%   MAKE_RESULTS_CHART() reads reports/metrics_summary.csv and writes
%   reports/results_chart.png — grouped bars for segmental SNR and
%   SI-SDR plus a STOI panel — so the README's chart is generated from
%   the same committed numbers as its table, never drawn by hand.

    rootDir = setup();
    summary = readtable(fullfile(rootDir, 'reports', 'metrics_summary.csv'));
    fig = figure('Visible', 'off', 'Position', [0 0 900 320]);
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile;
    bar(summary.SegSNR_dB);
    set(gca, 'XTickLabel', summary.Method);
    ylabel('Segmental SNR (dB)');
    title('Segmental SNR (higher is better)');
    grid on;
    nexttile;
    if any(strcmp(summary.Properties.VariableNames, 'SISDR_dB'))
        bar(summary.SISDR_dB);
        set(gca, 'XTickLabel', summary.Method);
        ylabel('SI-SDR (dB)');
        title('SI-SDR (higher is better)');
    else
        bar(summary.LSD_dB);
        set(gca, 'XTickLabel', summary.Method);
        ylabel('LSD (dB)');
        title('LSD (lower is better)');
    end
    grid on;
    nexttile;
    bar(summary.STOI);
    set(gca, 'XTickLabel', summary.Method);
    ylabel('STOI');
    ylim([0 1]);
    title('STOI (higher is better)');
    grid on;
    exportgraphics(fig, fullfile(rootDir, 'reports', 'results_chart.png'), ...
        'Resolution', 200);
    close(fig);
    fprintf('Wrote reports/results_chart.png\n');
end
