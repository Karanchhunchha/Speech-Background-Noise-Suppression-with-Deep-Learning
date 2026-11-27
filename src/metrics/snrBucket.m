function bucket = snrBucket(inputSnrDb)
%SNRBUCKET Assign a measured mixture SNR to the corpus's nominal level.
%   bucket = SNRBUCKET(inputSnrDb) returns, for each measured input
%   SNR (dB), the nearest of the VoiceBank-DEMAND test set's four
%   nominal mixture levels (2.5, 7.5, 12.5, 17.5 dB). The distributed
%   corpus files carry no per-file SNR label, so the level is
%   recovered from the clean reference: the measured mixture SNR of a
%   test file sits within a fraction of a dB of its nominal level,
%   which makes the nearest-level assignment unambiguous in practice
%   (the levels are 5 dB apart). Used for the per-SNR breakdown in
%   reports/metrics_by_snr.csv.

    levels = [2.5, 7.5, 12.5, 17.5];
    [~, idx] = min(abs(inputSnrDb(:) - levels), [], 2);
    bucket = levels(idx)';
    bucket = reshape(bucket, size(inputSnrDb));
end
