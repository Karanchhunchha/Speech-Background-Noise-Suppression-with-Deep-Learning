function [loss, gradients] = modelLossCRM(net, X, T)
%MODELLOSSCRM Masked-spectrum loss for the complex-mask network.
%   [loss, gradients] = MODELLOSSCRM(net, X, T) runs the dlnetwork
%   NET on a minibatch X (numFreqBins x segLen x 3 x N, dlarray
%   'SSCB': noisy real / specScale, noisy imaginary / specScale,
%   standardised log-magnitude), applies the predicted complex mask
%   (maskBound * tanh of the raw two-channel output, maskBound = 2)
%   to the noisy complex spectrogram carried in X's first two
%   channels, and scores the result against the clean complex
%   spectrogram T (numFreqBins x segLen x 2 x N, same scaling):
%
%       loss = MSE(est, clean) over the complex spectra
%              + 0.5 * MSE(|est|, |clean|) over magnitudes.
%
%   The magnitude term guards against over-suppression (the failure
%   mode that costs STOI): a mask that zeroes speech energy is
%   penalised even when its phase happens to be right. All targets
%   are precomputed spectra — no differentiable STFT is needed
%   inside the loss. Used with DLFEVAL in TRAIN_MODEL_V3's custom
%   training loop.

    Y = forward(net, X);
    mask = 2 * tanh(Y);
    mRe = mask(:, :, 1, :);
    mIm = mask(:, :, 2, :);
    noisyRe = X(:, :, 1, :);
    noisyIm = X(:, :, 2, :);
    estRe = mRe .* noisyRe - mIm .* noisyIm;
    estIm = mRe .* noisyIm + mIm .* noisyRe;
    cleanRe = T(:, :, 1, :);
    cleanIm = T(:, :, 2, :);
    lossComplex = mean((estRe - cleanRe) .^ 2 + (estIm - cleanIm) .^ 2, 'all');
    estMag = sqrt(estRe .^ 2 + estIm .^ 2);
    cleanMag = sqrt(cleanRe .^ 2 + cleanIm .^ 2);
    lossMag = mean((estMag - cleanMag) .^ 2, 'all');
    loss = lossComplex + 0.5 * lossMag;
    gradients = dlgradient(loss, net.Learnables);
end
