function download_data(varargin)
%DOWNLOAD_DATA Fetch the VoiceBank-DEMAND (16 kHz) data this project uses.
%   DOWNLOAD_DATA() downloads the standard test set (824 clean/noisy
%   pairs, two unseen speakers) and the first 800 training pairs of the
%   VoiceBank-DEMAND corpus from the Hugging Face mirror of the
%   Edinburgh DataShare release, and extracts them as .wav files:
%
%       data/voicebank/clean_test,  data/voicebank/noisy_test
%       data/voicebank/clean_train, data/voicebank/noisy_train
%
%   DOWNLOAD_DATA('ExtraTrainShards', 1) additionally downloads and
%   extracts training shard 1 (of shards 1-4; shard 0 is the default
%   one above). Extra shards add the corpus's remaining training
%   speakers — used by the Phase-1 hardening work for a genuinely
%   speaker-disjoint validation set and larger training runs. Every
%   extra shard is SHA-256 verified exactly like the default ones.
%   'MaxExtraPairs' caps how many pairs per extra shard are extracted
%   (default inf — the whole shard).
%
%   Dataset: C. Valentini-Botinhao, "Noisy speech database for training
%   speech enhancement algorithms and TTS models", Edinburgh DataShare,
%   2017. DOI 10.7488/ds/2117. Licence: CC BY 4.0. A 16 kHz mirror of
%   the DataShare release (JacobLinCool/VoiceBank-DEMAND-16k on
%   Hugging Face) is used because the scripts and metrics in this
%   project operate at 16 kHz.
%
%   Supply-chain safety: the download URL is pinned to one immutable
%   dataset revision, and every shard's SHA-256 is verified against
%   the values recorded below before anything is extracted. A shard
%   that fails the check is rejected with an error, never parsed.
%
%   The full corpus is NOT committed to this repository (size); only a
%   small sample under data/sample is, which is enough for the demo and
%   the unit tests. Run this function once before TRAIN_MODEL /
%   EVALUATE_MODELS for the full pipeline.

    p = inputParser;
    p.addParameter('ExtraTrainShards', [], @(x) isempty(x) || ...
        (isnumeric(x) && all(ismember(x, 1:4))));
    p.addParameter('MaxExtraPairs', inf, @(x) isscalar(x) && x > 0);
    p.parse(varargin{:});
    extraShards = p.Results.ExtraTrainShards;
    maxExtraPairs = p.Results.MaxExtraPairs;

    rootDir = setup();
    vbRoot = fullfile(rootDir, 'data', 'voicebank');
    testClean = fullfile(vbRoot, 'clean_test');
    testNoisy = fullfile(vbRoot, 'noisy_test');
    trainClean = fullfile(vbRoot, 'clean_train');
    trainNoisy = fullfile(vbRoot, 'noisy_train');

    % "Already present" means complete matched pairs in all four
    % folders — counting clean files alone would certify a
    % half-extracted corpus as done. Extra shards are fetched whenever
    % requested, regardless of this early return.
    basePresent = numel(listPairs(testClean, testNoisy)) >= 824 && ...
        numel(listPairs(trainClean, trainNoisy)) >= 800;
    if basePresent && isempty(extraShards)
        fprintf('VoiceBank-DEMAND data already present under %s\n', vbRoot);
        return;
    end

    dlDir = fullfile(rootDir, 'data', 'downloads');
    if ~exist(dlDir, 'dir'); mkdir(dlDir); end
    % Pinned dataset revision (immutable) + recorded SHA-256 per shard.
    revision = '4497db342d7312978c45690591fda86117831940';
    base = "https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/resolve/" ...
        + revision + "/data/";
    testSha256 = '08b43dbea4a15a5f3067233f5cb2de3e5c9a600a3ae5b969661740778f642c87';
    % SHA-256 of training shards 0-4, as listed by the dataset
    % repository's file metadata at the pinned revision.
    trainSha256 = { ...
        'd8195194a27e8d007502d6eacf5e0fac71adb9fe343ee228b78bae3559464319', ...
        'd8a5c932846399a7021b544157b25bfce736122f66a27239c13bbb9262aa98bc', ...
        'e486e0782f9c978cd2ea08fb25dbe1694cc4c87efaecde8b04308e683db8a91c', ...
        '6cf3cc86af272546cb94c1e8d26088b3a15feef1f662b145cea86a7c33387b6f', ...
        '2afbf89fc0b9e651f7990ff17d247ab18803d463842281a67c4a6b7c25abc7b2'};

    if ~basePresent
        testParquet = fullfile(dlDir, 'test.parquet');
        if ~exist(testParquet, 'file')
            fprintf('Downloading test shard (about 133 MB)...\n');
            websave(testParquet, base + "test-00000-of-00001.parquet");
        end
        verifySha256(testParquet, testSha256);
        nTest = extractParquetPairs(testParquet, testClean, testNoisy, inf);
        fprintf('Extracted %d test pairs.\n', nTest);

        trainParquet = fullfile(dlDir, 'train0.parquet');
        if ~exist(trainParquet, 'file')
            fprintf('Downloading training shard 1 of 5 (about 445 MB)...\n');
            websave(trainParquet, base + "train-00000-of-00005.parquet");
        end
        verifySha256(trainParquet, trainSha256{1});
        nTrain = extractParquetPairs(trainParquet, trainClean, trainNoisy, 800);
        fprintf('Extracted %d training pairs.\n', nTrain);
    end

    for shard = extraShards
        localName = sprintf('train%d.parquet', shard);
        shardParquet = fullfile(dlDir, localName);
        if ~exist(shardParquet, 'file')
            fprintf('Downloading extra training shard %d of 5 (about 420 MB)...\n', shard + 1);
            websave(shardParquet, base + sprintf("train-%05d-of-00005.parquet", shard));
        end
        verifySha256(shardParquet, trainSha256{shard + 1});
        nExtra = extractParquetPairs(shardParquet, trainClean, trainNoisy, maxExtraPairs);
        fprintf('Extracted %d pairs from extra shard %d.\n', nExtra, shard);
    end
end

function verifySha256(file, expected)
%VERIFYSHA256 Error unless FILE's SHA-256 equals EXPECTED (hex).
    actual = sha256OfFile(file);
    if ~strcmp(actual, expected)
        error('download_data:ChecksumMismatch', ...
            ['SHA-256 mismatch for %s.\nExpected %s\nActual   %s\n' ...
             'The file was NOT extracted. Delete it and re-run to fetch a fresh copy.'], ...
            file, expected, actual);
    end
    fprintf('SHA-256 verified for %s\n', file);
end

function hex = sha256OfFile(file)
%SHA256OFFILE SHA-256 of a file via the Java MessageDigest API.
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(file, 'r');
    if fid < 0
        error('download_data:ReadFailed', 'Cannot open %s.', file);
    end
    cleaner = onCleanup(@() fclose(fid));
    while ~feof(fid)
        chunk = fread(fid, 1024 * 1024, '*uint8');
        if ~isempty(chunk)
            md.update(chunk);
        end
    end
    digest = typecast(md.digest(), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2)', 1, []));
    clear cleaner;
end
