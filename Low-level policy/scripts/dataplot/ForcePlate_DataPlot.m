% ForcePlate_DataPlot.m
% Plot resultant force magnitude for 6 force-plate experiments.
%
% Data source:
%   logs/260506_Pitch_0.5hz_0.2amp_FP
%
% Expected column order in each TXT:
%   1 Fx, 2 Fy, 3 Fz, 4 Mx, 5 My, 6 Mz

clear; clc; close all;

%% User settings
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataDir = fullfile(repoRoot, 'logs', '260506_Pitch_0.5hz_0.2amp_FP');
sampleRateHz = 1000;
plotDurationS = 35.0;

% Set true if you want to subtract the initial static offset from Fx/Fy/Fz.
removeBaseline = false;
baselineSamples = 1000;

experiments = {
    'Dyna NoRand',    'Dyna_NoRand_0.5hz_0.2amp_FP.txt',       20.455
    'Dyna Rand',      'Dyna_Rand_0.5hz_0.2amp_FP.txt',         18.959
    'KinOnly NoRand', 'KinOnly_NoRand_0.5hz_0.2amp_FP.txt',    16.585
    'KinOnly Rand',   'KinOnly_Rand_0.5hz_0.2amp_FP.txt',      16.699
    'Prop NoRand',    'Prop_NoRand_0.5hz_0.2amp_FP.txt',       26.600
    'Prop Rand',      'Prop_Rand_0.5hz_0.2amp_FP.txt',         24.035
};

%% Load, compute vector magnitude, and plot
figure('Name', 'Force Plate Resultant Force Magnitude', 'Color', 'w');

if exist('tiledlayout', 'file') == 2
    tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    useTiledLayout = true;
else
    useTiledLayout = false;
end

fprintf('Force plate data directory: %s\n', dataDir);

for i = 1:size(experiments, 1)
    displayName = experiments{i, 1};
    fileName = experiments{i, 2};
    startTimeS = experiments{i, 3};
    filePath = fullfile(dataDir, fileName);

    data = loadForcePlateData(filePath);
    force = data(:, 1:3);
    [force, t] = extractTimeWindow(force, sampleRateHz, startTimeS, plotDurationS, filePath);

    if removeBaseline
        nBase = min(baselineSamples, size(force, 1));
        force = force - mean(force(1:nBase, :), 1, 'omitnan');
    end

    forceMagnitude = sqrt(sum(force.^2, 2));

    if useTiledLayout
        nexttile;
    else
        subplot(3, 2, i);
    end

    plot(t, forceMagnitude, 'LineWidth', 0.8);
    grid on;
    box on;
    title(sprintf('%s, start %.3f s', displayName, startTimeS), 'Interpreter', 'none');
    xlabel('Time from aligned start [s]');
    ylabel('|F| [N]');
    xlim([0, plotDurationS]);

    fprintf('%-16s start %.3f s, samples %d, plotted %.3f s, |F| mean %.6g, max %.6g\n', ...
        displayName, startTimeS, numel(forceMagnitude), t(end), ...
        mean(forceMagnitude, 'omitnan'), max(forceMagnitude, [], 'omitnan'));
end

if exist('sgtitle', 'file') == 2
    sgtitle(sprintf('Force plate resultant force magnitude, pitch 0.5 Hz / 0.2 amp, %.1f s window', plotDurationS), ...
        'Interpreter', 'none');
end

%% Local functions
function data = loadForcePlateData(filePath)
    if ~isfile(filePath)
        error('Force-plate data file not found: %s', filePath);
    end

    if exist('readmatrix', 'file') == 2
        data = readmatrix(filePath, 'FileType', 'text', 'Delimiter', ',');
    else
        data = dlmread(filePath, ','); %#ok<DLMRD>
    end

    % The exported TXT files can contain a leading blank line.
    data = data(any(~isnan(data), 2), :);

    if isempty(data) || size(data, 2) < 3
        error('Input file must contain at least Fx, Fy, Fz columns: %s', filePath);
    end

    data = data(:, 1:min(size(data, 2), 6));
end

function [windowData, tWindow] = extractTimeWindow(data, sampleRateHz, startTimeS, durationS, filePath)
    startIdx = round(startTimeS * sampleRateHz) + 1;
    nWindow = round(durationS * sampleRateHz);
    endIdx = startIdx + nWindow - 1;

    if startIdx < 1 || endIdx > size(data, 1)
        error('Requested window [%.3f, %.3f) s exceeds data length in %s.', ...
            startTimeS, startTimeS + durationS, filePath);
    end

    windowData = data(startIdx:endIdx, :);
    tWindow = (0:nWindow - 1).' ./ sampleRateHz;
end
