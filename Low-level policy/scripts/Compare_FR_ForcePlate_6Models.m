% Compare_FR_ForcePlate_6Models.m
% Compare simulated FR GRF sensor magnitude against force-plate magnitude.
%
% Sim data:
%   logs/rsl_rl/MCLrobotics_MCLQuadserial_attitude/
%     comparison_play_model1500_matched_ablation_amp0p2_freq0p5_60s
%     comparison_play_model1500_matched_ablation_amp0p2_freq0p5_40s
%
% Force plate data:
%   logs/260506_Pitch_0.5hz_0.2amp_FP
%
% Each subplot overlays:
%   - Sim FR ||F||
%   - ForcePlate ||F||

clear; clc; close all;

%% User settings
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

simLogDir60 = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p2_freq0p5_60s');
simPropRandLogDir = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p2_freq0p5_40s');

forcePlateDir = fullfile(repoRoot, 'logs', '260506_Pitch_0.5hz_0.2amp_FP');
forcePlateSampleRateHz = 1000;
compareDurationS = 20.0;
trimStartS = 0.1;
forcePlateShiftToSimS = 1.0;

% Keep false to compare raw exported force-plate vector magnitude.
removeForcePlateBaseline = false;
forcePlateBaselineSamples = 1000;

experiments = {
    % display name     sim CSV model          force-plate TXT                         FP start [s]
    'Dyna NoRand',     'pitch_Dyn_NoRand',    'Dyna_NoRand_0.5hz_0.2amp_FP.txt',       20.455
    'Dyna Rand',       'pitch_Dyn_Rand',      'Dyna_Rand_0.5hz_0.2amp_FP.txt',         18.959
    'KinOnly NoRand',  'pitch_Kinonly_NoRand','KinOnly_NoRand_0.5hz_0.2amp_FP.txt',    16.585
    'KinOnly Rand',    'pitch_Kinonly_Rand',  'KinOnly_Rand_0.5hz_0.2amp_FP.txt',      16.699
    'Prop NoRand',     'pitch_Prop_NoRand',   'Prop_NoRand_0.5hz_0.2amp_FP.txt',       26.600
    'Prop Rand',       'pitch_Prop_Rand',     'Prop_Rand_0.5hz_0.2amp_FP.txt',         24.035
};

%% Plot
figure('Name', 'FR GRF Sensor vs Force Plate Magnitude', 'Color', 'w');

if exist('tiledlayout', 'file') == 2
    tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    useTiledLayout = true;
else
    useTiledLayout = false;
end

fprintf('Sim 60s dir: %s\n', simLogDir60);
fprintf('Sim Prop Rand 40s dir: %s\n', simPropRandLogDir);
fprintf('ForcePlate dir: %s\n\n', forcePlateDir);

for i = 1:size(experiments, 1)
    displayName = experiments{i, 1};
    simModel = experiments{i, 2};
    forcePlateFile = experiments{i, 3};
    forcePlateStartS = experiments{i, 4};
    forcePlateAlignedStartS = forcePlateStartS + forcePlateShiftToSimS;

    if strcmp(simModel, 'pitch_Prop_Rand')
        simCsvPath = fullfile(simPropRandLogDir, [simModel, '.csv']);
    else
        simCsvPath = fullfile(simLogDir60, [simModel, '.csv']);
    end
    forcePlatePath = fullfile(forcePlateDir, forcePlateFile);

    [simT, simFRForce] = loadSimLegForceNorm(simCsvPath, 'FR', compareDurationS);
    [fpT, fpForce] = loadForcePlateMagnitude(forcePlatePath, forcePlateSampleRateHz, ...
        forcePlateAlignedStartS, compareDurationS, removeForcePlateBaseline, forcePlateBaselineSamples);
    [simT, simFRForce] = trimSignalStart(simT, simFRForce, trimStartS, 'sim', displayName);
    [fpT, fpForce] = trimSignalStart(fpT, fpForce, trimStartS, 'forceplate', displayName);
    fpForceAtSimT = interp1(fpT, fpForce, simT, 'linear', 'extrap');
    forceRmse = rmsOmitNan(simFRForce - fpForceAtSimT);

    if useTiledLayout
        nexttile;
    else
        subplot(3, 2, i);
    end

    plot(simT, simFRForce, 'LineWidth', 1.15, 'DisplayName', 'Sim FR ||F||');
    hold on;
    plot(fpT, fpForce, 'LineWidth', 1.0, 'DisplayName', 'ForcePlate ||F||');
    grid on;
    box on;
    xlim([0, compareDurationS - trimStartS]);
    xlabel('Time [s]');
    ylabel('|F| [N]');
    title(sprintf('%s, FP start %.3f + %.1f s', displayName, forcePlateStartS, forcePlateShiftToSimS), ...
        'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');

    fprintf('%-16s trim %.3f s, sim samples %5d, FP samples %5d, FP raw start %.3f s, aligned start %.3f s, RMSE %.3f N, sim mean %.3f, FP mean %.3f, sim max %.3f, FP max %.3f\n', ...
        displayName, trimStartS, numel(simFRForce), numel(fpForce), forcePlateStartS, forcePlateAlignedStartS, ...
        forceRmse, mean(simFRForce, 'omitnan'), mean(fpForce, 'omitnan'), ...
        max(simFRForce, [], 'omitnan'), max(fpForce, [], 'omitnan'));
end

if exist('sgtitle', 'file') == 2
    sgtitle(sprintf('Sim FR GRF sensor vs ForcePlate resultant force, %.1f s window, first %.1f s trimmed', ...
        compareDurationS, trimStartS), ...
        'Interpreter', 'none');
end

%% Local functions
function [t, forceNorm] = loadSimLegForceNorm(csvPath, legName, compareDurationS)
    if ~isfile(csvPath)
        error('Sim CSV not found: %s', csvPath);
    end

    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    vars = string(T.Properties.VariableNames);

    if ~ismember("time_s", vars)
        error('Missing time_s column in %s', csvPath);
    end

    sensorCols = [
        "Sensor_force_body_x_" + legName
        "Sensor_force_body_y_" + legName
        "Sensor_force_body_z_" + legName
    ];
    worldCols = [
        "W_grf_x_" + legName + "_foot"
        "W_grf_y_" + legName + "_foot"
        "W_grf_z_" + legName + "_foot"
    ];

    if all(ismember(sensorCols, vars))
        Fx = double(T.(sensorCols(1)));
        Fy = double(T.(sensorCols(2)));
        Fz = double(T.(sensorCols(3)));
    elseif all(ismember(worldCols, vars))
        warning('Sensor force columns missing in %s. Falling back to W_grf columns.', csvPath);
        Fx = double(T.(worldCols(1)));
        Fy = double(T.(worldCols(2)));
        Fz = double(T.(worldCols(3)));
    else
        error('FR force columns not found in %s.', csvPath);
    end

    tRaw = double(T.("time_s"));
    t = tRaw - tRaw(1);
    mask = t <= compareDurationS;

    if ~any(mask)
        error('No sim samples within %.3f s in %s.', compareDurationS, csvPath);
    end

    t = t(mask);
    forceNorm = sqrt(Fx(mask).^2 + Fy(mask).^2 + Fz(mask).^2);
end

function [t, forceMagnitude] = loadForcePlateMagnitude(filePath, sampleRateHz, startTimeS, durationS, removeBaseline, baselineSamples)
    data = loadForcePlateData(filePath);
    force = data(:, 1:3);
    [force, t] = extractTimeWindow(force, sampleRateHz, startTimeS, durationS, filePath);

    if removeBaseline
        nBase = min(baselineSamples, size(force, 1));
        force = force - mean(force(1:nBase, :), 1, 'omitnan');
    end

    forceMagnitude = sqrt(sum(force.^2, 2));
end

function data = loadForcePlateData(filePath)
    if ~isfile(filePath)
        error('Force-plate data file not found: %s', filePath);
    end

    if exist('readmatrix', 'file') == 2
        data = readmatrix(filePath, 'FileType', 'text', 'Delimiter', ',');
    else
        data = dlmread(filePath, ','); %#ok<DLMRD>
    end

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

function [tTrim, yTrim] = trimSignalStart(t, y, trimStartS, signalName, displayName)
    mask = t >= trimStartS;
    if ~any(mask)
        error('No %s samples remain after trimming %.3f s for %s.', ...
            signalName, trimStartS, displayName);
    end

    tTrim = t(mask);
    yTrim = y(mask);
    tTrim = tTrim - tTrim(1);
end

function y = rmsOmitNan(x)
    x = x(~isnan(x));
    if isempty(x)
        y = nan;
    else
        y = sqrt(mean(x.^2));
    end
end
