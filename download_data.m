function download_data()
%DOWNLOAD_DATA Fetch the VoiceBank-DEMAND (16 kHz) data this project uses.
%   DOWNLOAD_DATA() downloads the standard test set (824 clean/noisy
%   pairs, two unseen speakers) and the first 800 training pairs of the
%   VoiceBank-DEMAND corpus from the Hugging Face mirror of the
%   Edinburgh DataShare release, and extracts them as .wav files:
%
%       data/voicebank/clean_test,  data/voicebank/noisy_test
%       data/voicebank/clean_train, data/voicebank/noisy_train
%
%   Dataset: C. Valentini-Botinhao, "Noisy speech database for training
%   speech enhancement algorithms and TTS models", Edinburgh DataShare,
%   2017. DOI 10.7488/ds/2117. Licence: CC BY 4.0. The 16 kHz mirror
%   (JacobLinCool/VoiceBank-DEMAND-16k) is used because the scripts and
%   metrics in this project operate at 16 kHz; the content is identical
%   to the DataShare release resampled from 48 kHz.
%
%   The full corpus is NOT committed to this repository (size); only a
%   small sample under data/sample is, which is enough for the demo and
%   the unit tests. Run this function once before TRAIN_MODEL /
%   EVALUATE_MODELS for the full pipeline.

    rootDir = setup();
    vbRoot = fullfile(rootDir, 'data', 'voicebank');
    testClean = fullfile(vbRoot, 'clean_test');
    testNoisy = fullfile(vbRoot, 'noisy_test');
    trainClean = fullfile(vbRoot, 'clean_train');
    trainNoisy = fullfile(vbRoot, 'noisy_train');

    if numel(dir(fullfile(testClean, '*.wav'))) >= 800 && ...
       numel(dir(fullfile(trainClean, '*.wav'))) >= 100
        fprintf('VoiceBank-DEMAND data already present under %s\n', vbRoot);
        return;
    end

    dlDir = fullfile(rootDir, 'data', 'downloads');
    if ~exist(dlDir, 'dir'); mkdir(dlDir); end
    base = 'https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/resolve/main/data/';

    testParquet = fullfile(dlDir, 'test.parquet');
    if ~exist(testParquet, 'file')
        fprintf('Downloading test shard (about 133 MB)...\n');
        websave(testParquet, base + "test-00000-of-00001.parquet");
    end
    nTest = extractParquetPairs(testParquet, testClean, testNoisy, inf);
    fprintf('Extracted %d test pairs.\n', nTest);

    trainParquet = fullfile(dlDir, 'train0.parquet');
    if ~exist(trainParquet, 'file')
        fprintf('Downloading training shard 1 of 5 (about 445 MB)...\n');
        websave(trainParquet, base + "train-00000-of-00005.parquet");
    end
    nTrain = extractParquetPairs(trainParquet, trainClean, trainNoisy, 800);
    fprintf('Extracted %d training pairs.\n', nTrain);
end
