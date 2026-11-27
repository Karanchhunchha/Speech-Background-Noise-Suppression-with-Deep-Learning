function w = hannPeriodic(n)
%HANNPERIODIC Periodic Hann window of length n (toolbox-free).
%   w = HANNPERIODIC(n) returns the periodic Hann window used by the
%   STFT/ISTFT pair in this project. Implemented directly so the project
%   does not depend on the Signal Processing Toolbox window functions.

    if n == 1
        w = 1;
        return;
    end
    k = (0:n-1)';
    w = 0.5 - 0.5 * cos(2 * pi * k / n);
end
