function value = sisdrDb(reference, estimate)
%SISDRDB Scale-invariant signal-to-distortion ratio, in dB.
%   value = SISDRDB(reference, estimate) returns the scale-invariant
%   SDR (Le Roux et al., 2019) of ESTIMATE against the clean
%   REFERENCE: the estimate is projected onto the reference, and the
%   ratio of the projection's energy to the residual's energy is
%   reported in dB. Higher is better; the measure is invariant to a
%   constant gain on the estimate. Signals are truncated to the
%   shorter length and mean-removed, per the reference definition.

    reference = reference(:);
    estimate = estimate(:);
    n = min(numel(reference), numel(estimate));
    if n == 0
        error('sisdrDb:EmptyInput', 'Signals must be non-empty.');
    end
    reference = reference(1:n);
    estimate = estimate(1:n);
    reference = reference - mean(reference);
    estimate = estimate - mean(estimate);
    refEnergy = sum(reference .^ 2);
    if refEnergy == 0
        error('sisdrDb:ZeroReference', 'Reference signal has zero energy.');
    end
    alpha = sum(estimate .* reference) / refEnergy;
    projection = alpha * reference;
    residual = estimate - projection;
    resEnergy = sum(residual .^ 2);
    if resEnergy == 0
        value = Inf;
        return;
    end
    value = 10 * log10(sum(projection .^ 2) / resEnergy);
end
