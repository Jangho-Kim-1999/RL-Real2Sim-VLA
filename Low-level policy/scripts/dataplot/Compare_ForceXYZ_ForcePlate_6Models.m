% Compare_ForceXYZ_ForcePlate_6Models.m
% Compare sim FR contact-force xyz against force-plate xyz for 6 models.
%
% Default sim source is W_grf_x/y/z_FR_foot because force-plate axes are fixed.
% If you want body-frame sensor force instead, set simForceSource = 'sensor_body'.

clear; clc; close all;

%% User settings
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

simLogDir = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p1_freq1p2_wait2_36cycles_35s_total_xyz');
forcePlateDir = fullfile(repoRoot, 'logs', '260506_Pitch_0.5hz_0.2amp_FP');

compareDurationS = 35.0;
simStartS = 0.0;
forcePlateShiftToSimS = 0.0;  % new sim has 2 s wait, same as experiment/force plate
forcePlateSampleRateHz = 1000;
simLeg = 'FR';
simForceSource = 'W_grf';  % 'W_grf' or 'sensor_body'

% Adjust signs here if force-plate axis convention is opposite to sim.
simSign = [1, 1, 1];
forcePlateSign = [1, 1, 1];

experiments = {
    % display name      sim CSV model           force-plate TXT                         FP start [s]
    'Dyna NoRand',      'pitch_Dyn_NoRand',     'Dyna_NoRand_0.5hz_0.2amp_FP.txt',       20.455
    'Dyna Rand',        'pitch_Dyn_Rand',       'Dyna_Rand_0.5hz_0.2amp_FP.txt',         18.959
    'KinOnly NoRand',   'pitch_Kinonly_NoRand', 'KinOnly_NoRand_0.5hz_0.2amp_FP.txt',    16.585
    'KinOnly Rand',     'pitch_Kinonly_Rand',   'KinOnly_Rand_0.5hz_0.2amp_FP.txt',      16.699
    'Prop NoRand',      'pitch_Prop_NoRand',    'Prop_NoRand_0.5hz_0.2amp_FP.txt',       26.600
    'Prop Rand',        'pitch_Prop_Rand',      'Prop_Rand_0.5hz_0.2amp_FP.txt',         24.035
};

%% Plot
figure('Name', 'Sim FR Force XYZ vs ForcePlate XYZ', 'Color', 'w');
if exist('tiledlayout', 'file') == 2
    tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    useTiledLayout = true;
else
    useTiledLayout = false;
end

fprintf('Sim dir       : %s\n', simLogDir);
fprintf('ForcePlate dir: %s\n', forcePlateDir);
fprintf('Sim force source: %s, leg: %s\n\n', simForceSource, simLeg);

colors = lines(3);
axisNames = {'x', 'y', 'z'};

for i = 1:size(experiments, 1)
    displayName = experiments{i, 1};
    simModel = experiments{i, 2};
    forcePlateFile = experiments{i, 3};
    forcePlateStartS = experiments{i, 4};
    forcePlateAlignedStartS = forcePlateStartS + forcePlateShiftToSimS;

    simCsvPath = fullfile(simLogDir, [simModel, '.csv']);
    forcePlatePath = fullfile(forcePlateDir, forcePlateFile);

    [simT, simF] = loadSimForceXYZ(simCsvPath, simLeg, simForceSource, simStartS, compareDurationS);
    [fpT, fpF] = loadForcePlateXYZ(forcePlatePath, forcePlateSampleRateHz, forcePlateAlignedStartS, compareDurationS);
    simF = simF .* simSign;
    fpF = fpF .* forcePlateSign;

    fpAtSimT = interp1(fpT, fpF, simT, 'linear', 'extrap');
    rmse = sqrt(mean((simF - fpAtSimT).^2, 1, 'omitnan'));

    if useTiledLayout
        nexttile;
    else
        subplot(3, 2, i);
    end

    hold on;
    for k = 1:3
        plot(simT, simF(:, k), '-', 'Color', colors(k, :), 'LineWidth', 1.10, ...
            'DisplayName', ['sim F' axisNames{k}]);
        plot(fpT, fpF(:, k), '--', 'Color', colors(k, :), 'LineWidth', 0.95, ...
            'DisplayName', ['FP F' axisNames{k}]);
    end
    grid on;
    box on;
    xlim([0, compareDurationS]);
    xlabel('Time [s]');
    ylabel('Force [N]');
    title(sprintf('%s, FP start %.3f s', displayName, forcePlateAlignedStartS), 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');

    fprintf('%-16s samples sim %5d FP %5d, RMSE Fx %.3f N, Fy %.3f N, Fz %.3f N\n', ...
        displayName, size(simF, 1), size(fpF, 1), rmse(1), rmse(2), rmse(3));
end

if exist('sgtitle', 'file') == 2
    sgtitle(sprintf('Sim %s %s force xyz vs ForcePlate xyz, %.1f s window', ...
        simLeg, simForceSource, compareDurationS), 'Interpreter', 'none');
end

%% Local functions
function [t, F] = loadSimForceXYZ(csvPath, leg, source, startS, durationS)
    if ~isfile(csvPath)
        error('Sim CSV not found: %s', csvPath);
    end
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    vars = string(T.Properties.VariableNames);

    if strcmp(source, 'sensor_body')
        cols = [
            "Sensor_force_body_x_" + leg
            "Sensor_force_body_y_" + leg
            "Sensor_force_body_z_" + leg
        ];
    else
        cols = [
            "W_grf_x_" + leg + "_foot"
            "W_grf_y_" + leg + "_foot"
            "W_grf_z_" + leg + "_foot"
        ];
    end
    if ~all(ismember(cols, vars))
        error('Missing sim force columns in %s: %s', csvPath, strjoin(cols(~ismember(cols, vars)), ', '));
    end

    tRaw = getTableColumn(T, 'time_s');
    tRel = tRaw - tRaw(1);
    mask = tRel >= startS & tRel <= startS + durationS;
    if ~any(mask)
        error('No sim force samples in [%.3f, %.3f] s from %s.', startS, startS + durationS, csvPath);
    end
    t = tRel(mask) - startS;
    Fx = double(T.(char(cols(1))));
    Fy = double(T.(char(cols(2))));
    Fz = double(T.(char(cols(3))));
    F = [Fx(mask), Fy(mask), Fz(mask)];
end

function [t, F] = loadForcePlateXYZ(filePath, sampleRateHz, startS, durationS)
    data = loadForcePlateData(filePath);
    startIdx = round(startS * sampleRateHz) + 1;
    nWindow = round(durationS * sampleRateHz);
    endIdx = startIdx + nWindow - 1;
    if startIdx < 1 || endIdx > size(data, 1)
        error('Requested force-plate window [%.3f, %.3f) s exceeds data length in %s.', ...
            startS, startS + durationS, filePath);
    end
    F = data(startIdx:endIdx, 1:3);
    t = (0:nWindow - 1).' ./ sampleRateHz;
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
end

function x = getTableColumn(T, requestedName)
    vars = string(T.Properties.VariableNames);
    idx = find(strcmp(strtrim(vars), requestedName), 1);
    if isempty(idx)
        error('Missing column: %s', requestedName);
    end
    x = double(T.(T.Properties.VariableNames{idx}));
end
