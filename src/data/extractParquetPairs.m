function count = extractParquetPairs(parquetFile, cleanDir, noisyDir, maxPairs)
%EXTRACTPARQUETPAIRS Extract clean/noisy wavs from a VoiceBank parquet.
%   count = EXTRACTPARQUETPAIRS(parquetFile, cleanDir, noisyDir, maxPairs)
%   reads a parquet shard of the JacobLinCool/VoiceBank-DEMAND-16k
%   dataset (columns: id, clean, noisy, where clean/noisy are audio
%   structs with a bytes field holding the .wav content) and writes up
%   to maxPairs pairs as .wav files into the two folders. Returns the
%   number of pairs written.

    if ~exist(cleanDir, 'dir'); mkdir(cleanDir); end
    if ~exist(noisyDir, 'dir'); mkdir(noisyDir); end
    T = parquetread(parquetFile);
    ids = string(T.id);
    cleanBytes = audioBytesColumn(T.clean);
    noisyBytes = audioBytesColumn(T.noisy);
    count = min([maxPairs, height(T), numel(ids)]);
    for i = 1:count
        writeBytes(fullfile(cleanDir, ids(i) + ".wav"), cleanBytes{i});
        writeBytes(fullfile(noisyDir, ids(i) + ".wav"), noisyBytes{i});
    end
end

function bytesList = audioBytesColumn(col)
%AUDIOBYTESCOLUMN Normalise a parquet audio column to a cell of uint8.
%   The audio feature is a struct column with fields bytes/path; after
%   PARQUETREAD it arrives either as a table with a bytes variable or
%   as a struct array. Handle both shapes.
    if istable(col)
        bytesList = col.bytes;
    elseif isstruct(col)
        bytesList = {col.bytes};
        bytesList = bytesList(:);
    else
        error('extractParquetPairs:UnexpectedAudioColumn', ...
            'Unexpected audio column type: %s', class(col));
    end
    if ~iscell(bytesList)
        bytesList = num2cell(bytesList, 2);
    end
end

function writeBytes(file, bytes)
    fid = fopen(file, 'w');
    if fid < 0
        error('extractParquetPairs:WriteFailed', 'Cannot open %s for writing.', file);
    end
    fwrite(fid, bytes, 'uint8');
    fclose(fid);
end
