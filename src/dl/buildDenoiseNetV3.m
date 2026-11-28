function lgraph = buildDenoiseNetV3(numFreqBins, segLen)
%BUILDENOISENETV3 Residual dilated CNN for complex ratio masking.
%   lgraph = BUILDENOISENETV3(numFreqBins, segLen) returns the layer
%   graph of the project's second-generation network. Where the
%   shipped model (BUILDENOISENET) predicts a real-valued magnitude
%   mask and reuses the noisy phase, this network predicts a COMPLEX
%   ratio mask (two output channels: real and imaginary parts), so
%   the phase is corrected as well — the magnitude-only design's
%   known ceiling on STOI and PESQ.
%
%   Inputs are numFreqBins x segLen x 3 segments: the noisy STFT's
%   real and imaginary parts (scaled by the model's specScale) plus
%   the standardised log-magnitude, so the network sees both phase
%   and level. The body is four residual blocks (48 channels) whose
%   3x3 convolutions are dilated along TIME by 1, 2, 4 and 8 — a
%   dilated-convolution (TCN-style) receptive field of several
%   hundred frames at a fraction of the cost of recurrent layers,
%   which are awkward and slow to train on spectrogram graphs.
%   About 180k parameters: still CPU-trainable, deliberately far
%   below the multi-million-parameter research models.
%
%   The graph has no output layer: training uses a custom loop with
%   MODELLOSSCRM (masked-spectrum loss), and inference applies
%   maskBound * tanh(output) in DENOISEWITHNET.

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, imageInputLayer( ...
        [numFreqBins segLen 3], 'Normalization', 'none', 'Name', 'input'));
    stem = [ ...
        convolution2dLayer(3, 48, 'Padding', 'same', 'Name', 'stem_conv')
        batchNormalizationLayer('Name', 'stem_bn')
        reluLayer('Name', 'stem_relu')];
    lgraph = addLayers(lgraph, stem);
    lgraph = connectLayers(lgraph, 'input', 'stem_conv');

    prev = 'stem_relu';
    dilations = [1 2 4 8];
    for b = 1:numel(dilations)
        d = dilations(b);
        pfx = sprintf('blk%d', b);
        main = [ ...
            convolution2dLayer(3, 48, 'Padding', 'same', ...
                'DilationFactor', [1 d], 'Name', [pfx '_conv1'])
            batchNormalizationLayer('Name', [pfx '_bn1'])
            reluLayer('Name', [pfx '_relu1'])
            convolution2dLayer(3, 48, 'Padding', 'same', ...
                'DilationFactor', [1 d], 'Name', [pfx '_conv2'])
            batchNormalizationLayer('Name', [pfx '_bn2'])];
        lgraph = addLayers(lgraph, main);
        lgraph = addLayers(lgraph, additionLayer(2, 'Name', [pfx '_add']));
        lgraph = addLayers(lgraph, reluLayer('Name', [pfx '_relu2']));
        lgraph = connectLayers(lgraph, prev, [pfx '_conv1']);
        lgraph = connectLayers(lgraph, [pfx '_bn2'], [pfx '_add/in1']);
        lgraph = connectLayers(lgraph, prev, [pfx '_add/in2']);
        lgraph = connectLayers(lgraph, [pfx '_add'], [pfx '_relu2']);
        prev = [pfx '_relu2'];
    end

    head = [ ...
        convolution2dLayer(3, 24, 'Padding', 'same', 'Name', 'head_conv')
        batchNormalizationLayer('Name', 'head_bn')
        reluLayer('Name', 'head_relu')
        convolution2dLayer(3, 2, 'Padding', 'same', 'Name', 'mask')];
    lgraph = addLayers(lgraph, head);
    lgraph = connectLayers(lgraph, prev, 'head_conv');
end
