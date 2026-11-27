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
end
