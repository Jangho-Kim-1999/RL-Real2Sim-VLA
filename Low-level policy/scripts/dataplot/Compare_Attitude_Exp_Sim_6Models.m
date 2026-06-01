% Compare_Attitude_Exp_Sim_6Models.m
% Compare real experiment body attitude against sim body attitude.
%
% Experiment data:
%   logs/260506_Pitch_0.5hz_0.2amp
%
% Sim data:
%   logs/rsl_rl/MCLrobotics_MCLQuadserial_attitude/
%     comparison_play_model1500_matched_ablation_amp0p2_freq0p5_60s
%     comparison_play_model1500_matched_ablation_amp0p2_freq0p5_40s
%
% Real data uses body_rot_0/1/2 when available. If direct roll/pitch are not
% available, roll/pitch are reconstructed from policy_projected_gravity_0/1/2.

clear; clc; close all;

%% User settings
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

expLogDir = fullfile(repoRoot, 'logs', '260506_Pitch_0.5hz_0.2amp');
simLogDir60 = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p2_freq0p5_60s');
simPropRandLogDir = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p2_freq0p5_40s');

compareDurationS = 15.0;
plotDegrees = false;
expShiftToSimS = 1.0;

% Absolute yaw has an arbitrary heading offset between real and sim.
% Keep this true to compare yaw drift instead of absolute heading.
zeroInitialYaw = true;

experiments = {
    % display name      sim CSV model           experiment CSV                         exp start [s]
    'Dyna NoRand',      'pitch_Dyn_NoRand',     'Dyna_NoRand_0.5hz_0.2amp.csv',        0.0
    'Dyna Rand',        'pitch_Dyn_Rand',       'Dyna_Rand_0.5hz_0.2amp.csv',          0.0
    'KinOnly NoRand',   'pitch_Kinonly_NoRand', 'KinOnly_NoRand_0.5hz_0.2amp.csv',     0.0
    'KinOnly Rand',     'pitch_Kinonly_Rand',   'KinOnly_Rand_0.5hz_0.2amp.csv',       0.0
    'Prop NoRand',      'pitch_Prop_NoRand',    'Prop_NoRand_0.5hz_0.2amp.csv',        0.0
    'Prop Rand',        'pitch_Prop_Rand',      'Prop_Rand_0.5hz_0.2amp.csv',          0.0
};

%% Plot
figure('Name', 'Experiment vs Sim Body Attitude', 'Color', 'w');

if exist('tiledlayout', 'file') == 2
    tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    useTiledLayout = true;
else
    useTiledLayout = false;
end

fprintf('Experiment dir: %s\n', expLogDir);
fprintf('Sim 60s dir   : %s\n', simLogDir60);
fprintf('Sim PropRand  : %s\n\n', simPropRandLogDir);

for i = 1:size(experiments, 1)
    displayName = experiments{i, 1};
    simModel = experiments{i, 2};
    expFile = experiments{i, 3};
    expStartS = experiments{i, 4};
    expAlignedStartS = expStartS + expShiftToSimS;

    if strcmp(simModel, 'pitch_Prop_Rand')
        simCsvPath = fullfile(simPropRandLogDir, [simModel, '.csv']);
    else
        simCsvPath = fullfile(simLogDir60, [simModel, '.csv']);
    end
    expCsvPath = fullfile(expLogDir, expFile);

    [simT, simRPY] = loadSimRPY(simCsvPath, compareDurationS, zeroInitialYaw);
    [expT, expRPY, expSource] = loadExperimentRPY(expCsvPath, expAlignedStartS, compareDurationS, zeroInitialYaw);

    actualDurationS = min([compareDurationS, simT(end), expT(end)]);
    simMask = simT <= actualDurationS;
    expMask = expT <= actualDurationS;
    simT = simT(simMask);
    simRPY = simRPY(simMask, :);
    expT = expT(expMask);
    expRPY = expRPY(expMask, :);

    expAtSimT = interp1(expT, expRPY, simT, 'linear', 'extrap');
    rmse = sqrt(mean((simRPY - expAtSimT).^2, 1, 'omitnan'));

    if plotDegrees
        simRPYPlot = rad2deg(simRPY);
        expRPYPlot = rad2deg(expRPY);
        rmsePrint = rad2deg(rmse);
        yLabelText = 'Angle [deg]';
        unitText = 'deg';
    else
        simRPYPlot = simRPY;
        expRPYPlot = expRPY;
        rmsePrint = rmse;
        yLabelText = 'Angle [rad]';
        unitText = 'rad';
    end

    if useTiledLayout
        nexttile;
    else
        subplot(3, 2, i);
    end

    hold on;
    plot(simT, simRPYPlot(:, 1), 'r-',  'LineWidth', 1.15, 'DisplayName', 'sim roll');
    plot(expT, expRPYPlot(:, 1), 'r--', 'LineWidth', 1.00, 'DisplayName', 'exp roll');
    plot(simT, simRPYPlot(:, 2), 'b-',  'LineWidth', 1.15, 'DisplayName', 'sim pitch');
    plot(expT, expRPYPlot(:, 2), 'b--', 'LineWidth', 1.00, 'DisplayName', 'exp pitch');
    plot(simT, simRPYPlot(:, 3), 'k-',  'LineWidth', 1.15, 'DisplayName', 'sim yaw');
    plot(expT, expRPYPlot(:, 3), 'k--', 'LineWidth', 1.00, 'DisplayName', 'exp yaw');
    grid on;
    box on;
    xlim([0, actualDurationS]);
    xlabel('Time [s]');
    ylabel(yLabelText);
    title(sprintf('%s, exp start %.3f + %.1f s', displayName, expStartS, expShiftToSimS), ...
        'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');

    fprintf('%-16s source %-18s duration %.3f s, samples sim %5d exp %5d, exp raw start %.3f s, aligned start %.3f s, RMSE roll %.5f %s, pitch %.5f %s, yaw %.5f %s\n', ...
        displayName, expSource, actualDurationS, size(simRPY, 1), size(expRPY, 1), expStartS, expAlignedStartS, ...
        rmsePrint(1), unitText, rmsePrint(2), unitText, rmsePrint(3), unitText);
end

if exist('sgtitle', 'file') == 2
    if zeroInitialYaw
        yawText = 'yaw zeroed at initial value';
    else
        yawText = 'absolute yaw';
    end
    sgtitle(sprintf('Experiment vs Sim roll/pitch/yaw, %.1f s window, exp shifted %.1f s, %s', ...
        compareDurationS, expShiftToSimS, yawText), ...
        'Interpreter', 'none');
end

%% Local functions
function [t, rpy] = loadSimRPY(csvPath, compareDurationS, zeroInitialYaw)
    if ~isfile(csvPath)
        error('Sim CSV not found: %s', csvPath);
    end

    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    roll = getTableColumn(T, 'base_roll');
    pitch = getTableColumn(T, 'base_pitch');
    yaw = unwrap(getTableColumn(T, 'base_yaw'));
    tRaw = getTableColumn(T, 'time_s');

    t = tRaw - tRaw(1);
    mask = t <= compareDurationS & isfinite(t) & isfinite(roll) & isfinite(pitch) & isfinite(yaw);
    if ~any(mask)
        error('No valid sim attitude samples within %.3f s in %s.', compareDurationS, csvPath);
    end

    t = t(mask);
    rpy = [roll(mask), pitch(mask), yaw(mask)];
    if zeroInitialYaw
        rpy(:, 3) = rpy(:, 3) - rpy(1, 3);
    end
end

function [t, rpy, sourceText] = loadExperimentRPY(csvPath, startTimeS, compareDurationS, zeroInitialYaw)
    if ~isfile(csvPath)
        error('Experiment CSV not found: %s', csvPath);
    end

    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    tRaw = getTableColumn(T, 't');
    tRel = tRaw - tRaw(1);

    if hasTableColumn(T, 'body_rot_0') && hasTableColumn(T, 'body_rot_1')
        roll = getTableColumn(T, 'body_rot_0');
        pitch = getTableColumn(T, 'body_rot_1');
        if hasTableColumn(T, 'body_rot_2')
            yaw = unwrap(getTableColumn(T, 'body_rot_2'));
        else
            yaw = nan(size(roll));
        end
        sourceText = 'body_rot';
    elseif hasTableColumn(T, 'policy_projected_gravity_0') && ...
            hasTableColumn(T, 'policy_projected_gravity_1') && ...
            hasTableColumn(T, 'policy_projected_gravity_2')
        gx = getTableColumn(T, 'policy_projected_gravity_0');
        gy = getTableColumn(T, 'policy_projected_gravity_1');
        gz = getTableColumn(T, 'policy_projected_gravity_2');
        gx = max(min(gx, 1), -1);
        roll = atan2(-gy, -gz);
        pitch = asin(gx);
        yaw = nan(size(roll));
        sourceText = 'projected_gravity';
    else
        error('No body_rot_* or policy_projected_gravity_* columns found in %s.', csvPath);
    end

    tEndS = startTimeS + compareDurationS;
    mask = tRel >= startTimeS & tRel <= tEndS & ...
        isfinite(tRel) & isfinite(roll) & isfinite(pitch);
    if ~any(mask)
        error('No valid experiment attitude samples in [%.3f, %.3f] s from %s.', ...
            startTimeS, tEndS, csvPath);
    end

    t = tRel(mask) - startTimeS;
    rpy = [roll(mask), pitch(mask), yaw(mask)];

    if zeroInitialYaw && any(isfinite(rpy(:, 3)))
        rpy(:, 3) = rpy(:, 3) - rpy(find(isfinite(rpy(:, 3)), 1, 'first'), 3);
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

function tf = hasTableColumn(T, requestedName)
    vars = string(T.Properties.VariableNames);
    tf = any(strcmp(strtrim(vars), requestedName));
end
