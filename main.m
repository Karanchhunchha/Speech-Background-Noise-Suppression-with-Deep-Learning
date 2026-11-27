function main(varargin)
%MAIN Single entry point — run the whole system end to end.
%   MAIN() runs the complete pipeline with minimal manual steps:
%     1. the unit-test suite (verifies every component),
%     2. the listening demo (before/after audio + spectrograms),
%     3. the objective evaluation (segmental SNR, LSD, STOI).
%   The evaluation uses the full VoiceBank-DEMAND test set when it has
%   been downloaded (see DOWNLOAD_DATA) and the committed sample pairs
%   otherwise, so MAIN works on a fresh clone with no downloads.
%
%   Name-value options:
%     'Full'      download the corpus and train the CNN from scratch
%                 before evaluating (about an hour; default false)
%     'Stage'     run one stage only: 'test', 'demo', 'evaluate',
%                 'train' (download + train), or 'all' (default)
%     'MaxFiles'  cap the evaluation at this many files (default Inf)

    p = inputParser;
    addParameter(p, 'Full', false);
    addParameter(p, 'Stage', 'all');
    addParameter(p, 'MaxFiles', Inf);
    parse(p, varargin{:});
    stage = lower(string(p.Results.Stage));

    if p.Results.Full || stage == "train"
        download_data();
        train_model();
        if stage == "train"
            return;
        end
    end

    if stage == "all" || stage == "test"
        ok = run_all_tests();
        if ~ok
            error('main:TestsFailed', ...
                'The unit-test suite did not pass; aborting the pipeline.');
        end
    end
    if stage == "all" || stage == "demo"
        demo_denoise();
    end
    if stage == "all" || stage == "evaluate"
        if isinf(p.Results.MaxFiles)
            evaluate_models();
        else
            evaluate_models('MaxFiles', p.Results.MaxFiles);
        end
    end
    if ~(stage == "all" || stage == "test" || stage == "demo" || ...
            stage == "evaluate" || stage == "train")
        error('main:BadStage', 'Unknown stage "%s".', stage);
    end
    fprintf('\nMAIN: pipeline finished (stage: %s).\n', stage);
end
