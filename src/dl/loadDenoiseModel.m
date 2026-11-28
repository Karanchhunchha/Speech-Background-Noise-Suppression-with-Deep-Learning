function model = loadDenoiseModel(modelFile)
%LOADDENOISEMODEL Load a trained denoising model artifact.
%   model = LOADDENOISEMODEL(modelFile) loads the .mat artifact written
%   by TRAIN_MODEL: a struct with fields net (the trained network),
%   mu and sigma (log-magnitude standardisation constants from the
%   training data) and winLen, hop, segLen (the STFT configuration).

    data = load(modelFile);
    if ~isfield(data, 'model')
        error('loadDenoiseModel:BadArtifact', ...
            'File %s does not contain a denoising model.', modelFile);
    end
    model = data.model;
    required = {'net', 'mu', 'sigma', 'winLen', 'hop', 'segLen'};
    for k = 1:numel(required)
        if ~isfield(model, required{k})
            error('loadDenoiseModel:BadArtifact', ...
                'Model in %s is missing the field "%s".', modelFile, required{k});
        end
    end
    if ~isfield(model, 'kind')
        % Artifacts written before the complex-mask model existed are
        % magnitude-mask (IRM) models.
        model.kind = 'irm';
    end
    if strcmp(model.kind, 'crm') && ~isfield(model, 'specScale')
        error('loadDenoiseModel:BadArtifact', ...
            'Complex-mask model in %s is missing the field "specScale".', modelFile);
    end
end
