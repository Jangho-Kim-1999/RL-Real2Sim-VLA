function plot_attitude_6models_model1500(logDir60, propRandLogDir, compareUntilS)
% plot_attitude_6models_model1500
%
% Compare 6 attitude-play logs. By default, 5 models are loaded from the
% 60s result folder and Prop Rand is loaded from the separate 40s folder.
% All models are cropped to the first 40 seconds for a fair comparison.
%
% Figure 1:
%   - subplot 1: roll reference and base_roll for all 6 models
%   - subplot 2: pitch reference and base_pitch for all 6 models
%
% Figure 2:
%   - 4 subplots, one per leg
%   - each subplot compares 6 models' GRF sensor force norm
%
% Usage:
%   plot_attitude_6models_model1500
%   plot_attitude_6models_model1500(logDir60)
%   plot_attitude_6models_model1500(logDir60, propRandLogDir)
%   plot_attitude_6models_model1500(logDir60, propRandLogDir, compareUntilS)

repoRoot = fileparts(fileparts(mfilename('fullpath')));
defaultLogDir60 = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p2_freq0p5_60s');
defaultPropRandLogDir = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p2_freq0p5_40s');

if nargin < 1 || isempty(logDir60) || strlength(string(logDir60)) == 0
    logDir60 = defaultLogDir60;
end
if nargin < 2 || isempty(propRandLogDir) || strlength(string(propRandLogDir)) == 0
    propRandLogDir = defaultPropRandLogDir;
end
if nargin < 3 || isempty(compareUntilS)
    compareUntilS = 40.0;
end

models = {
    'pitch_Dyn_NoRand'
    'pitch_Dyn_Rand'
    'pitch_Kinonly_NoRand'
    'pitch_Kinonly_Rand'
    'pitch_Prop_NoRand'
    'pitch_Prop_Rand'
};

displayNames = {
    'Dyn NoRand'
    'Dyn Rand'
    'Kinonly NoRand'
    'Kinonly Rand'
    'Prop NoRand'
    'Prop Rand'
};

csvDirs = repmat({char(logDir60)}, size(models));
propRandIdx = find(strcmp(models, 'pitch_Prop_Rand'), 1);
csvDirs{propRandIdx} = char(propRandLogDir);

legs = {'FL', 'FR', 'RL', 'RR'};
colors = lines(numel(models));
data = struct([]);

for i = 1:numel(models)
    csvPath = fullfile(csvDirs{i}, [models{i}, '.csv']);
    if ~isfile(csvPath)
        error('CSV not found: %s', csvPath);
    end

    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    vars = string(T.Properties.VariableNames);

    requireColumns(vars, ["time_s", "cmd_roll_ref", "cmd_pitch_ref", "base_roll", "base_pitch"], csvPath);

    tRaw = double(T.("time_s"));
    t = tRaw - tRaw(1);
    sourceDurationS = t(end);
    cropMask = t <= compareUntilS;
    if ~any(cropMask)
        error('No samples within %.3f s in %s', compareUntilS, csvPath);
    end
    if sourceDurationS < compareUntilS
        warning('CSV shorter than compareUntilS: %s ends at %.3f s, requested %.3f s.', ...
            csvPath, sourceDurationS, compareUntilS);
    end
    t = t(cropMask);
    rollRef = double(T.("cmd_roll_ref"));
    pitchRef = double(T.("cmd_pitch_ref"));
    baseRoll = double(T.("base_roll"));
    basePitch = double(T.("base_pitch"));

    data(i).model = models{i}; %#ok<AGROW>
    data(i).displayName = displayNames{i};
    data(i).csvPath = csvPath;
    data(i).csvDir = csvDirs{i};
    data(i).t = t;
    data(i).durationS = t(end);
    data(i).sourceDurationS = sourceDurationS;
    data(i).numSamples = numel(t);
    data(i).rollRef = rollRef(cropMask);
    data(i).pitchRef = pitchRef(cropMask);
    data(i).baseRoll = baseRoll(cropMask);
    data(i).basePitch = basePitch(cropMask);
    data(i).rollRmse = rmsOmitNan(data(i).baseRoll - data(i).rollRef);
    data(i).pitchRmse = rmsOmitNan(data(i).basePitch - data(i).pitchRef);

    for j = 1:numel(legs)
        grfNorm = readLegGrfNorm(T, vars, legs{j});
        data(i).grf.(legs{j}) = grfNorm(cropMask);
    end
end

plotTrackingFigure(data, colors, compareUntilS);
plotGrfFigure(data, colors, legs, compareUntilS);
printMetrics(data, legs, compareUntilS);

end

function plotTrackingFigure(data, colors, compareUntilS)
figure('Name', 'Command Tracking: Roll/Pitch, 6 Models', 'Color', 'w');
tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
hold on; grid on;
plot(data(1).t, data(1).rollRef, 'k--', 'LineWidth', 1.8, 'DisplayName', 'roll ref');
for i = 1:numel(data)
    name = sprintf('%s (RMSE %.4f)', data(i).displayName, data(i).rollRmse);
    plot(data(i).t, data(i).baseRoll, 'LineWidth', 1.25, 'Color', colors(i,:), 'DisplayName', name);
end
ylabel('Roll [rad]');
xlim([0, compareUntilS]);
title(sprintf('Roll Tracking, first %.1f s', compareUntilS));
legend('Location', 'best', 'Interpreter', 'none');

nexttile;
hold on; grid on;
plot(data(1).t, data(1).pitchRef, 'k--', 'LineWidth', 1.8, 'DisplayName', 'pitch ref');
for i = 1:numel(data)
    name = sprintf('%s (RMSE %.4f)', data(i).displayName, data(i).pitchRmse);
    plot(data(i).t, data(i).basePitch, 'LineWidth', 1.25, 'Color', colors(i,:), 'DisplayName', name);
end
xlabel('Time [s]');
ylabel('Pitch [rad]');
xlim([0, compareUntilS]);
title(sprintf('Pitch Tracking, first %.1f s', compareUntilS));
legend('Location', 'best', 'Interpreter', 'none');
end

function plotGrfFigure(data, colors, legs, compareUntilS)
figure('Name', 'GRF Sensor Force Norm: 6 Models', 'Color', 'w');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for j = 1:numel(legs)
    leg = legs{j};
    nexttile;
    hold on; grid on;
    for i = 1:numel(data)
        y = data(i).grf.(leg);
        plot(data(i).t, y, 'LineWidth', 1.15, 'Color', colors(i,:), 'DisplayName', data(i).displayName);
    end
    title(sprintf('%s GRF Sensor ||F||, first %.1f s', leg, compareUntilS), 'Interpreter', 'none');
    xlabel('Time [s]');
    ylabel('Force [N]');
    xlim([0, compareUntilS]);
    legend('Location', 'best', 'Interpreter', 'none');
end
end

function Fnorm = readLegGrfNorm(T, vars, leg)
% Prefer body-frame sensor force. Fall back to world-frame W_grf columns.
sensorCols = [
    "Sensor_force_body_x_" + leg
    "Sensor_force_body_y_" + leg
    "Sensor_force_body_z_" + leg
];
worldCols = [
    "W_grf_x_" + leg + "_foot"
    "W_grf_y_" + leg + "_foot"
    "W_grf_z_" + leg + "_foot"
];

if all(ismember(sensorCols, vars))
    Fx = double(T.(sensorCols(1)));
    Fy = double(T.(sensorCols(2)));
    Fz = double(T.(sensorCols(3)));
elseif all(ismember(worldCols, vars))
    Fx = double(T.(worldCols(1)));
    Fy = double(T.(worldCols(2)));
    Fz = double(T.(worldCols(3)));
else
    warning('GRF columns missing for %s. Returning NaN.', leg);
    Fnorm = nan(height(T), 1);
    return;
end

Fnorm = sqrt(Fx.^2 + Fy.^2 + Fz.^2);
end

function requireColumns(vars, required, csvPath)
missing = required(~ismember(required, vars));
if ~isempty(missing)
    error('Missing columns in %s: %s', csvPath, strjoin(missing, ', '));
end
end

function y = rmsOmitNan(x)
x = x(~isnan(x));
if isempty(x)
    y = nan;
else
    y = sqrt(mean(x.^2));
end
end

function printMetrics(data, legs, compareUntilS)
fprintf('\n=== Command tracking RMSE, first %.3f s ===\n', compareUntilS);
for i = 1:numel(data)
    fprintf('%-18s source %.3f s, plotted %.3f s, samples %d, roll %.6f rad, pitch %.6f rad\n', ...
        data(i).displayName, data(i).sourceDurationS, data(i).durationS, data(i).numSamples, ...
        data(i).rollRmse, data(i).pitchRmse);
    fprintf('  csv: %s\n', data(i).csvPath);
end

fprintf('\n=== GRF sensor force norm summary [N], first %.3f s ===\n', compareUntilS);
for i = 1:numel(data)
    fprintf('%s\n', data(i).displayName);
    for j = 1:numel(legs)
        leg = legs{j};
        y = data(i).grf.(leg);
        fprintf('  %-2s mean %.3f, max %.3f, nonzero %d/%d\n', leg, mean(y, 'omitnan'), max(y, [], 'omitnan'), nnz(y > 0), numel(y));
    end
end
end
