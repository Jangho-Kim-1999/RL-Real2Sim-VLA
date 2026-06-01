% Compare_1p2hz_0p1amp_SimExp_All.m
% Unified comparison for the 1.2 Hz / 0.1 rad pitch experiment:
%   1) Sim FR world-frame GRF xyz vs force-plate xyz
%   2) Sim pitch angle vs experiment pitch angle
%   3) Sim 12 joint angles vs experiment 12 joint angles
%   4) Sim 12 joint torques vs experiment 12 actuator torques
%
% Notes:
% - Sim W_grf_* columns are IsaacLab ContactSensor normal force vectors in
%   world frame. On flat ground, x/y are expected to be near zero and z is
%   the normal GRF. For body-frame rotated normal force, set simForceSource
%   to 'sensor_body', but that is not a world-frame force-plate comparison.
% - Force-plate files are assumed to store Fx,Fy,Fz in columns 1:3.

clear; clc; close all;

%% User settings
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

simLogDir = fullfile(repoRoot, 'logs', 'rsl_rl', 'MCLrobotics_MCLQuadserial_attitude', ...
    'comparison_play_model1500_matched_ablation_amp0p1_freq1p2_wait2_36cycles_35s');
expLogDir = fullfile(repoRoot, 'logs', '260506_Pitch_1.2hz_0.1amp');
forcePlateDir = fullfile(repoRoot, 'logs', '260506_Pitch_1.2hz_0.1amp_FP');

compareDurationS = 35.0;
simStartS = 0.0;
expShiftToSimS = 0.0;       % sim and exp both use 2 s initial wait for this run
forcePlateShiftToSimS = 0.0;
forcePlateSampleRateHz = 1000;

plotDegrees = false;
zeroInitialPitch = false;

% World-frame force comparison uses W_grf. Set 'sensor_body' only if you
% intentionally want body-frame rotated normal force, not world-frame force.
simLeg = 'FR';
simForceSource = 'W_grf';   % 'W_grf' or 'sensor_body'
simForceSign = [1, 1, 1];
forcePlateSign = [1, 1, 1];
removeForcePlateBias = false;
forcePlateBiasWindowS = 1.0;

% Sim torque column prefix can be changed to 'tau_no_comp_' or 'tau_comp_' if needed.
simTorquePrefix = 'torque_';
expTorquePrefix = 'actuator_torque_';

jointNames = {
    'FLHAA', 'FLHIP', 'FLKNEE', ...
    'FRHAA', 'FRHIP', 'FRKNEE', ...
    'RLHAA', 'RLHIP', 'RLKNEE', ...
    'RRHAA', 'RRHIP', 'RRKNEE'
};

experiments = {
    % display name      sim CSV model           exp CSV                              force-plate TXT                         exp start  FP start
    'Dyna NoRand',      'pitch_Dyn_NoRand',     'Dyna_NoRand_1.2hz_0.1amp.csv',      'Dyna_NoRand_1.2hz_0.1amp_FP.txt',       0.0,       0.0
    'Dyna Rand',        'pitch_Dyn_Rand',       'Dyna_Rand_1.2hz_0.1amp.csv',        'Dyna_Rand_1.2hz_0.1amp_FP.txt',         0.0,       0.0
    'KinOnly NoRand',   'pitch_Kinonly_NoRand', 'KinOnly_NoRand_1.2hz_0.1amp.csv',   'KinOnly_NoRand_1.2hz_0.1amp_FP.txt',    0.0,       0.0
    'KinOnly Rand',     'pitch_Kinonly_Rand',   'KinOnly_Rand_1.2hz_0.1amp.csv',     'KinOnly_Rand_1.2hz_0.1amp_FP.txt',      0.0,       0.0
    'Prop NoRand',      'pitch_Prop_NoRand',    'Prop_NoRand_1.2hz_0.1amp.csv',      'Prop_NoRand_1.2hz_0.1amp_FP.txt',       0.0,       0.0
    'Prop Rand',        'pitch_Prop_Rand',      'Prop_Rand_1.2hz_0.1amp.csv',        'Prop_Rand_1.2hz_0.1amp_FP.txt',         0.0,       0.0
};

fprintf('Sim dir       : %s\n', simLogDir);
fprintf('Exp dir       : %s\n', expLogDir);
fprintf('ForcePlate dir: %s\n', forcePlateDir);
fprintf('Compare window: %.3f s\n', compareDurationS);
fprintf('Sim force     : %s, leg %s\n', simForceSource, simLeg);
fprintf('Sim torque    : %s*\n\n', simTorquePrefix);

%% Load all data once
nExp = size(experiments, 1);
D = struct([]);

for i = 1:nExp
    D(i).displayName = experiments{i, 1};
    D(i).simModel = experiments{i, 2};
    D(i).expFile = experiments{i, 3};
    D(i).fpFile = experiments{i, 4};
    D(i).expStartS = experiments{i, 5} + expShiftToSimS;
    D(i).fpStartS = experiments{i, 6} + forcePlateShiftToSimS;

    simCsvPath = fullfile(simLogDir, [D(i).simModel, '.csv']);
    expCsvPath = fullfile(expLogDir, D(i).expFile);
    fpPath = fullfile(forcePlateDir, D(i).fpFile);

    [D(i).simT, D(i).simForce] = loadSimForceXYZ(simCsvPath, simLeg, simForceSource, simStartS, compareDurationS);
    D(i).simForce = D(i).simForce .* simForceSign;

    [D(i).fpT, D(i).fpForce] = loadForcePlateXYZ(fpPath, forcePlateSampleRateHz, D(i).fpStartS, ...
        compareDurationS, removeForcePlateBias, forcePlateBiasWindowS);
    D(i).fpForce = D(i).fpForce .* forcePlateSign;

    [D(i).simPitchT, D(i).simPitch] = loadSimPitch(simCsvPath, simStartS, compareDurationS, zeroInitialPitch);
    [D(i).expPitchT, D(i).expPitch, D(i).expPitchSource] = loadExperimentPitch(expCsvPath, D(i).expStartS, ...
        compareDurationS, zeroInitialPitch);

    [D(i).simJointT, D(i).simQ] = loadSimJointMatrix(simCsvPath, jointNames, 'joint_angle_', simStartS, compareDurationS);
    [D(i).expJointT, D(i).expQ] = loadExpJointMatrix(expCsvPath, 'joint_angle_', D(i).expStartS, compareDurationS);

    [D(i).simTorqueT, D(i).simTau] = loadSimJointMatrix(simCsvPath, jointNames, simTorquePrefix, simStartS, compareDurationS);
    [D(i).expTorqueT, D(i).expTau] = loadExpJointMatrix(expCsvPath, expTorquePrefix, D(i).expStartS, compareDurationS);
end

%% Figure 1: GRF xyz, 6 subplots
figure('Name', '1.2 Hz 0.1 amp: Sim FR GRF xyz vs ForcePlate xyz', 'Color', 'w');
useTiledLayout = startTiledLayout(3, 2);
forceColors = lines(3);
axisNames = {'x', 'y', 'z'};

fprintf('=== Force RMSE: sim %s %s vs force plate xyz [N] ===\n', simLeg, simForceSource);
for i = 1:nExp
    selectTile(useTiledLayout, 3, 2, i);
    hold on;
    for k = 1:3
        plot(D(i).simT, D(i).simForce(:, k), '-', 'Color', forceColors(k, :), 'LineWidth', 1.10, ...
            'DisplayName', ['sim F' axisNames{k}]);
        plot(D(i).fpT, D(i).fpForce(:, k), '--', 'Color', forceColors(k, :), 'LineWidth', 0.95, ...
            'DisplayName', ['FP F' axisNames{k}]);
    end
    actualDurationS = min([compareDurationS, D(i).simT(end), D(i).fpT(end)]);
    xlim([0, actualDurationS]);
    grid on; box on;
    xlabel('Time [s]'); ylabel('Force [N]');
    title(sprintf('%s, FP start %.3f s', D(i).displayName, D(i).fpStartS), 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');

    fpAtSimT = interp1(D(i).fpT, D(i).fpForce, D(i).simT, 'linear', 'extrap');
    rmse = calcRmse(D(i).simForce, fpAtSimT);
    fprintf('%-16s RMSE Fx %.3f, Fy %.3f, Fz %.3f | sim max abs [%.3g %.3g %.3g]\n', ...
        D(i).displayName, rmse(1), rmse(2), rmse(3), max(abs(D(i).simForce), [], 1));
end
addSuperTitle(sprintf('Sim %s %s world/contact force xyz vs force plate xyz', simLeg, simForceSource));
fprintf('\n');

%% Figure 2: pitch angle, 6 subplots
figure('Name', '1.2 Hz 0.1 amp: Pitch angle sim vs experiment', 'Color', 'w');
useTiledLayout = startTiledLayout(3, 2);
fprintf('=== Pitch RMSE ===\n');
for i = 1:nExp
    selectTile(useTiledLayout, 3, 2, i);
    actualDurationS = min([compareDurationS, D(i).simPitchT(end), D(i).expPitchT(end)]);
    simMask = D(i).simPitchT <= actualDurationS;
    expMask = D(i).expPitchT <= actualDurationS;
    simT = D(i).simPitchT(simMask);
    expT = D(i).expPitchT(expMask);
    simPitch = D(i).simPitch(simMask);
    expPitch = D(i).expPitch(expMask);

    expAtSimT = interp1(expT, expPitch, simT, 'linear', 'extrap');
    rmse = calcRmse(simPitch, expAtSimT);

    if plotDegrees
        simPlot = rad2deg(simPitch);
        expPlot = rad2deg(expPitch);
        rmsePrint = rad2deg(rmse);
        yLabelText = 'Pitch [deg]';
        unitText = 'deg';
    else
        simPlot = simPitch;
        expPlot = expPitch;
        rmsePrint = rmse;
        yLabelText = 'Pitch [rad]';
        unitText = 'rad';
    end

    plot(simT, simPlot, 'b-', 'LineWidth', 1.15, 'DisplayName', 'sim pitch');
    hold on;
    plot(expT, expPlot, 'b--', 'LineWidth', 1.00, 'DisplayName', 'exp pitch');
    grid on; box on;
    xlim([0, actualDurationS]);
    xlabel('Time [s]'); ylabel(yLabelText);
    title(sprintf('%s RMSE %.5f %s', D(i).displayName, rmsePrint, unitText), 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none');
    fprintf('%-16s source %-12s RMSE %.6f %s\n', D(i).displayName, D(i).expPitchSource, rmsePrint, unitText);
end
addSuperTitle('Pitch angle: sim base\_pitch vs experiment body\_rot\_1');
fprintf('\n');

%% Figures 3-8: joint angles per model, 12 subplots each
fprintf('=== Joint angle RMSE ===\n');
for i = 1:nExp
    [simT, simQ, expT, expQ, actualDurationS] = alignPair(D(i).simJointT, D(i).simQ, D(i).expJointT, D(i).expQ, compareDurationS);
    expAtSimT = interp1(expT, expQ, simT, 'linear', 'extrap');
    rmse = calcRmse(simQ, expAtSimT);

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

    figure('Name', ['1.2 Hz 0.1 amp: Joint angles - ' D(i).displayName], 'Color', 'w');
    useTiledLayout = startTiledLayout(3, 4);
    for j = 1:numel(jointNames)
        selectTile(useTiledLayout, 3, 4, j);
        plot(simT, simPlot(:, j), 'LineWidth', 1.10, 'DisplayName', 'sim');
        hold on;
        plot(expT, expPlot(:, j), '--', 'LineWidth', 0.95, 'DisplayName', 'exp');
        grid on; box on;
        xlim([0, actualDurationS]);
        xlabel('Time [s]'); ylabel(yLabelText);
        title(sprintf('%s RMSE %.4f %s', jointNames{j}, rmsePrint(j), unitText), 'Interpreter', 'none');
        legend('Location', 'best', 'Interpreter', 'none');
    end
    addSuperTitle(sprintf('%s joint angles: sim vs exp', D(i).displayName));
    fprintf('%s joint angle RMSE [%s]\n', D(i).displayName, unitText);
    printJointRmse(jointNames, rmsePrint);
end
fprintf('\n');

%% Figures 9-14: joint torques per model, 12 subplots each
fprintf('=== Joint torque RMSE ===\n');
for i = 1:nExp
    [simT, simTau, expT, expTau, actualDurationS] = alignPair(D(i).simTorqueT, D(i).simTau, ...
        D(i).expTorqueT, D(i).expTau, compareDurationS);
    expAtSimT = interp1(expT, expTau, simT, 'linear', 'extrap');
    rmse = calcRmse(simTau, expAtSimT);

    figure('Name', ['1.2 Hz 0.1 amp: Joint torques - ' D(i).displayName], 'Color', 'w');
    useTiledLayout = startTiledLayout(3, 4);
    for j = 1:numel(jointNames)
        selectTile(useTiledLayout, 3, 4, j);
        plot(simT, simTau(:, j), 'LineWidth', 1.10, 'DisplayName', 'sim');
        hold on;
        plot(expT, expTau(:, j), '--', 'LineWidth', 0.95, 'DisplayName', 'exp');
        grid on; box on;
        xlim([0, actualDurationS]);
        xlabel('Time [s]'); ylabel('Torque [Nm]');
        title(sprintf('%s RMSE %.3f Nm', jointNames{j}, rmse(j)), 'Interpreter', 'none');
        legend('Location', 'best', 'Interpreter', 'none');
    end
    addSuperTitle(sprintf('%s joint torques: sim %s vs exp %s', D(i).displayName, simTorquePrefix, expTorquePrefix));
    fprintf('%s joint torque RMSE [Nm]\n', D(i).displayName);
    printJointRmse(jointNames, rmse);
end

fprintf('\nDone. Adjust expStartS/fpStartS in the experiments table if raw logs need manual alignment.\n');

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

function [t, F] = loadForcePlateXYZ(filePath, sampleRateHz, startS, durationS, removeBias, biasWindowS)
    data = loadForcePlateData(filePath);
    startIdx = round(startS * sampleRateHz) + 1;
    nWindow = round(durationS * sampleRateHz);
    endIdx = startIdx + nWindow - 1;
    if startIdx < 1 || endIdx > size(data, 1)
        error('Requested force-plate window [%.3f, %.3f) s exceeds data length in %s.', ...
            startS, startS + durationS, filePath);
    end
    F = data(startIdx:endIdx, 1:3);
    if removeBias
        nBias = max(1, min(round(biasWindowS * sampleRateHz), size(F, 1)));
        F = F - mean(F(1:nBias, :), 1, 'omitnan');
    end
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

function [t, pitch] = loadSimPitch(csvPath, startS, durationS, zeroInitialPitch)
    if ~isfile(csvPath)
        error('Sim CSV not found: %s', csvPath);
    end
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    tRaw = getTableColumn(T, 'time_s');
    tRel = tRaw - tRaw(1);
    pitchRaw = getTableColumn(T, 'base_pitch');
    mask = tRel >= startS & tRel <= startS + durationS & isfinite(pitchRaw);
    if ~any(mask)
        error('No valid sim pitch samples in [%.3f, %.3f] from %s.', startS, startS + durationS, csvPath);
    end
    t = tRel(mask) - startS;
    pitch = pitchRaw(mask);
    if zeroInitialPitch
        pitch = pitch - pitch(1);
    end
end

function [t, pitch, sourceText] = loadExperimentPitch(csvPath, startS, durationS, zeroInitialPitch)
    if ~isfile(csvPath)
        error('Experiment CSV not found: %s', csvPath);
    end
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    tRaw = getTableColumn(T, 't');
    tRel = tRaw - tRaw(1);

    if hasTableColumn(T, 'body_rot_1')
        pitchRaw = getTableColumn(T, 'body_rot_1');
        sourceText = 'body_rot_1';
    elseif hasTableColumn(T, 'policy_projected_gravity_0')
        gx = max(min(getTableColumn(T, 'policy_projected_gravity_0'), 1), -1);
        pitchRaw = asin(gx);
        sourceText = 'projected_gravity';
    else
        error('No body_rot_1 or policy_projected_gravity_0 column found in %s.', csvPath);
    end

    mask = tRel >= startS & tRel <= startS + durationS & isfinite(pitchRaw);
    if ~any(mask)
        error('No valid experiment pitch samples in [%.3f, %.3f] from %s.', startS, startS + durationS, csvPath);
    end
    t = tRel(mask) - startS;
    pitch = pitchRaw(mask);
    if zeroInitialPitch
        pitch = pitch - pitch(1);
    end
end

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

function [simT, simX, expT, expX, actualDurationS] = alignPair(simTIn, simXIn, expTIn, expXIn, requestedDurationS)
    actualDurationS = min([requestedDurationS, simTIn(end), expTIn(end)]);
    simMask = simTIn <= actualDurationS;
    expMask = expTIn <= actualDurationS;
    simT = simTIn(simMask);
    simX = simXIn(simMask, :);
    expT = expTIn(expMask);
    expX = expXIn(expMask, :);
end

function rmse = calcRmse(A, B)
    diff = A - B;
    rmse = sqrt(mean(diff.^2, 1, 'omitnan'));
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

function useTiledLayout = startTiledLayout(nRows, nCols)
    if exist('tiledlayout', 'file') == 2
        tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');
        useTiledLayout = true;
    else
        useTiledLayout = false;
    end
end

function selectTile(useTiledLayout, nRows, nCols, idx)
    if useTiledLayout
        nexttile;
    else
        subplot(nRows, nCols, idx);
    end
end

function addSuperTitle(titleText)
    if exist('sgtitle', 'file') == 2
        sgtitle(titleText, 'Interpreter', 'none');
    end
end

function printJointRmse(jointNames, rmse)
    for j = 1:numel(jointNames)
        fprintf('  %-7s %.6f\n', jointNames{j}, rmse(j));
    end
end
