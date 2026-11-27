function layers = buildDenoiseNet(numFreqBins, segLen)
%BUILDENOISENET Compact convolutional mask-estimation network.
%   layers = BUILDENOISENET(numFreqBins, segLen) returns the layer graph
%   of a small fully-convolutional network that maps a standardised
%   log-magnitude spectrogram segment (numFreqBins x segLen) to an
%   ideal-ratio-mask estimate of the same size. The network is
%   deliberately small (about 24k parameters) so it trains on CPU in
%   minutes: five 3x3 convolutions with batch normalisation and ReLU,
%   and a linear regression head (mask values are clipped to [0, 1] at
%   inference time in DENOISEWITHNET).

    layers = [
        imageInputLayer([numFreqBins segLen 1], 'Normalization', 'none', 'Name', 'input')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv4')
        batchNormalizationLayer('Name', 'bn4')
        reluLayer('Name', 'relu4')
        convolution2dLayer(3, 1, 'Padding', 'same', 'Name', 'mask')
        regressionLayer('Name', 'output')];
end
