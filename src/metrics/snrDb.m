function v = snrDb(clean, est)
%SNRDB Global signal-to-noise ratio of an estimate against clean speech.
%   v = SNRDB(clean, est) returns 10*log10(||clean||^2 / ||clean-est||^2)
%   in dB. The signals are truncated to the shorter length. A perfect
%   estimate returns Inf.

    n = min(numel(clean), numel(est));
    c = clean(1:n);
    e = est(1:n);
    err = c - e;
    v = 10 * log10(sum(c .^ 2) / max(sum(err .^ 2), eps));
end
