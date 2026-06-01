% Compare_JointAngles_Exp_Sim_6Models.m
% For each of 6 models, plot 12 joint-angle subplots comparing sim vs exp.

clear; clc; close all;

%% User settings
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
simLogDir = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p2_freq0p5_wait2_cmd60_62s');
expLogDir = fullfile(repoRoot, 'logs', '260506_Pitch_0.5hz_0.2amp');

compareDurationS = 35.0;
simStartS = 0.0;
expShiftToSimS = 0.0;  % new sim has 2 s wait, same as experiment
plotDegrees = false;

jointNames = {
    'FLHAA', 'FLHIP', 'FLKNEE', ...
    'FRHAA', 'FRHIP', 'FRKNEE', ...
    'RLHAA', 'RLHIP', 'RLKNEE', ...
    'RRHAA', 'RRHIP', 'RRKNEE'
};

experiments = {
    % display name      sim CSV model           experiment CSV                       exp start [s]
    'Dyna NoRand',      'pitch_Dyn_NoRand',     'Dyna_NoRand_0.5hz_0.2amp.csv',      0.0
    'Dyna Rand',        'pitch_Dyn_Rand',       'Dyna_Rand_0.5hz_0.2amp.csv',        0.0
    'KinOnly NoRand',   'pitch_Kinonly_NoRand', 'KinOnly_NoRand_0.5hz_0.2amp.csv',   0.0
    'KinOnly Rand',     'pitch_Kinonly_Rand',   'KinOnly_Rand_0.5hz_0.2amp.csv',     0.0
    'Prop NoRand',      'pitch_Prop_NoRand',    'Prop_NoRand_0.5hz_0.2amp.csv',      0.0
    'Prop Rand',        'pitch_Prop_Rand',      'Prop_Rand_0.5hz_0.2amp.csv',        0.0
};

fprintf('Sim dir: %s\n', simLogDir);
fprintf('Exp dir: %s\n\n', expLogDir);

for i = 1:size(experiments, 1)
    displayName = experiments{i, 1};
    simModel = experiments{i, 2};
    expFile = experiments{i, 3};
    expStartS = experiments{i, 4} + expShiftToSimS;

    simCsvPath = fullfile(simLogDir, [simModel, '.csv']);
    expCsvPath = fullfile(expLogDir, expFile);

    [simT, simQ] = loadSimJointMatrix(simCsvPath, jointNames, 'joint_angle_', simStartS, compareDurationS);
    [expT, expQ] = loadExpJointMatrix(expCsvPath, 'joint_angle_', expStartS, compareDurationS);

    actualDurationS = min([compareDurationS, simT(end), expT(end)]);
    simMask = simT <= actualDurationS;
    expMask = expT <= actualDurationS;
    simT = simT(simMask);
    simQ = simQ(simMask, :);
    expT = expT(expMask);
    expQ = expQ(expMask, :);

    expAtSimT = interp1(expT, expQ, simT, 'linear', 'extrap');
    rmse = sqrt(mean((simQ - expAtSimT).^2, 1, 'omitnan'));

    if plotDegrees
        simPlot = rad2deg(simQ);
        expPlot = rad2deg(expQ);
        rmsePrint = rad2deg(rmse);
        yLabelText = 'Joint angle [deg]';
        unitText = 'deg';
    else
        simPlot = simQ;
        expPlot = expQ;
        rmsePrint = rmse;
        yLabelText = 'Joint angle [rad]';
        unitText = 'rad';
    end

    figure('Name', ['Joint Angles: ' displayName], 'Color', 'w');
    if exist('tiledlayout', 'file') == 2
        tiledlayout(3, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
        useTiledLayout = true;
    else
        useTiledLayout = false;
    end

    for j = 1:numel(jointNames)
        if useTiledLayout
            nexttile;
        else
            subplot(3, 4, j);
        end
        plot(simT, simPlot(:, j), 'LineWidth', 1.10, 'DisplayName', 'sim');
        hold on;
        plot(expT, expPlot(:, j), '--', 'LineWidth', 0.95, 'DisplayName', 'exp');
        grid on;
        box on;
        xlim([0, actualDurationS]);
        title(sprintf('%s RMSE %.4f %s', jointNames{j}, rmsePrint(j), unitText), 'Interpreter', 'none');
        xlabel('Time [s]');
        ylabel(yLabelText);
        legend('Location', 'best', 'Interpreter', 'none');
    end

    if exist('sgtitle', 'file') == 2
        sgtitle(sprintf('%s joint angle sim vs exp, %.1f s window', displayName, actualDurationS), ...
            'Interpreter', 'none');
    end

    fprintf('%s joint angle RMSE [%s]\n', displayName, unitText);
    printJointRmse(jointNames, rmsePrint);
end

%% Local functions
function [t, X] = loadSimJointMatrix(csvPath, jointNames, prefix, startS, durationS)
    if ~isfile(csvPath)
        error('Sim CSV not found: %s', csvPath);
    end
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    tRaw = getTableColumn(T, 'time_s');
    tRel = tRaw - tRaw(1);
    mask = tRel >= startS & tRel <= startS + durationS;
    if ~any(mask)
        error('No sim samples in [%.3f, %.3f] s from %s.', startS, startS + durationS, csvPath);
    end
    t = tRel(mask) - startS;
    X = nan(nnz(mask), numel(jointNames));
    for j = 1:numel(jointNames)
        vals = getTableColumn(T, [prefix jointNames{j}]);
        X(:, j) = vals(mask);
    end
end

function [t, X] = loadExpJointMatrix(csvPath, prefix, startS, durationS)
    if ~isfile(csvPath)
        error('Experiment CSV not found: %s', csvPath);
    end
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    tRaw = getTableColumn(T, 't');
    tRel = tRaw - tRaw(1);
    mask = tRel >= startS & tRel <= startS + durationS;
    if ~any(mask)
        error('No exp samples in [%.3f, %.3f] s from %s.', startS, startS + durationS, csvPath);
    end
    t = tRel(mask) - startS;
    X = nan(nnz(mask), 12);
    for j = 1:12
        col = sprintf('%s%d', prefix, j - 1);
        vals = getTableColumn(T, col);
        X(:, j) = vals(mask);
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

function printJointRmse(jointNames, rmse)
    for j = 1:numel(jointNames)
        fprintf('  %-7s %.6f\n', jointNames{j}, rmse(j));
    end
end
