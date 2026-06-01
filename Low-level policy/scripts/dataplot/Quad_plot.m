clear; clc; close all;

%% User settings
filename = '260429_exp/260429_Proposed_1.5ms.csv';
plot_time_start_s = 0.0;
plot_time_end_s = inf;
plot_x_limits = [];  % Example: [0 1]. Use [] for automatic x-limits.

%% Data load
[~, dataset_label, ~] = fileparts(filename);
T = readtable(filename, 'VariableNamingRule', 'preserve');
raw = table2array(T);
vars = string(T.Properties.VariableNames);

num_joints = 12;
expected_cols = 1 + num_joints * 4;

if isempty(raw)
    error('Quad_plot:EmptyData', 'data.csv is empty.');
end

if size(raw, 2) < expected_cols
    error('Quad_plot:ColumnMismatch', ...
        'Expected at least %d columns in data.csv, got %d. Regenerate data.csv with the updated logger.', ...
        expected_cols, size(raw, 2));
end

t_all = raw(:, 1);
if isinf(plot_time_end_s)
    idx = t_all >= plot_time_start_s;
else
    idx = (t_all >= plot_time_start_s) & (t_all <= plot_time_end_s);
end

if ~any(idx)
    error('Quad_plot:TimeRange', 'No samples found in the requested time range.');
end

data = raw(idx, :);
T = T(idx, :);
vars = string(T.Properties.VariableNames);
t = data(:, 1);

col = 2;
joint_angle = data(:, col:col + num_joints - 1);
col = col + num_joints;

ctrl_input = data(:, col:col + num_joints - 1);
col = col + num_joints;

actuator_torque = data(:, col:col + num_joints - 1);
col = col + num_joints;

joint_ref = data(:, col:col + num_joints - 1);

%% Plot parameters
lw = 1.2;
ref_lw = 1.8;
title_size = 14;
axis_size = 11;
legend_size = 9;

set(0, 'defaultAxesFontName', 'Times New Roman');
set(0, 'defaultTextFontName', 'Times New Roman');

leg_names = {'FL', 'FR', 'RL', 'RR'};
joint_names = {'HAA', 'HIP', 'KNEE'};
joint_labels = cell(1, num_joints);
joint_axis_labels = cell(1, num_joints);

for leg = 1:4
    for joint = 1:3
        joint_idx = (leg - 1) * 3 + joint;
        joint_labels{joint_idx} = sprintf('%s-%s', leg_names{leg}, joint_names{joint});
        joint_axis_labels{joint_idx} = sprintf('%s%s', leg_names{leg}, joint_names{joint});
    end
end

%% Joint angles and references
plotJointTypeOverlay(1, t, joint_ref, joint_angle, leg_names, joint_names, ref_lw, lw, ...
    plot_x_limits, ...
    legend_size, axis_size, dataset_label, 'Joint Angle');

%% Actuator torque
plotJointTypeSingle(2, t, actuator_torque, leg_names, joint_names, lw, axis_size, ...
    plot_x_limits, 'Actuator Torque');

%% Control input
plotJointTypeSingle(3, t, ctrl_input, leg_names, joint_names, lw, axis_size, ...
    plot_x_limits, 'Control Input');

%% Policy observation plots
fig_num = 4;
fig_num = plotStackedGroupIfPresent(t, T, vars, 'body_rot_rate_', fig_num, ...
    plot_x_limits, ...
    'Body Angular Velocity', 'Angular Velocity');
fig_num = plotStackedGroupIfPresent(t, T, vars, 'body_rot_', fig_num, ...
    plot_x_limits, ...
    'Body Rotation', 'Angle');
fig_num = plotStackedGroupIfPresent(t, T, vars, 'policy_base_ang_vel_', fig_num, ...
    plot_x_limits, ...
    'Policy Base Angular Velocity', 'Angular Velocity');
fig_num = plotStackedGroupIfPresent(t, T, vars, 'policy_projected_gravity_', fig_num, ...
    plot_x_limits, ...
    'Policy Projected Gravity', 'Gravity');
fig_num = plotStackedGroupIfPresent(t, T, vars, 'policy_velocity_command_', fig_num, ...
    plot_x_limits, ...
    'Policy Velocity Command', 'Command');
fig_num = plotJointGroupIfPresent(t, T, vars, 'policy_joint_vel_rel_', fig_num, ...
    lw, axis_size, string(joint_axis_labels), plot_x_limits, ...
    'Policy Joint Velocity Relative');

setFigurePositions(4)

function plotJointTypeOverlay(fig_num, t, ref_data, actual_data, leg_names, joint_names, ref_lw, lw, x_limits, legend_size, axis_size, dataset_label, fig_title)
n_joint_types = numel(joint_names);
axs = ensureStackedAxes(fig_num, n_joint_types, fig_title, joint_names, axis_size, '', x_limits);
colors = lines(numel(leg_names));

for joint_idx = 1:n_joint_types
    ax = axs(joint_idx);
    hold(ax, 'on');
    for leg_idx = 1:numel(leg_names)
        data_idx = (leg_idx - 1) * n_joint_types + joint_idx;
        color = colors(leg_idx, :);
        plot(ax, t, actual_data(:, data_idx), 'LineWidth', lw, 'Color', color, ...
            'DisplayName', sprintf('%s %s', leg_names{leg_idx}, dataset_label));
        plot(ax, t, ref_data(:, data_idx), '--', 'LineWidth', ref_lw, 'Color', color, ...
            'DisplayName', sprintf('%s Ref', leg_names{leg_idx}));
    end

    if joint_idx == 1
        lgd = legend(ax, 'show');
        lgd.FontSize = legend_size;
        lgd.Interpreter = 'none';
        lgd.Location = 'best';
        lgd.AutoUpdate = 'on';
    end
end
end

function plotJointTypeSingle(fig_num, t, data_arr, leg_names, joint_names, lw, axis_size, x_limits, fig_title)
n_joint_types = numel(joint_names);
axs = ensureStackedAxes(fig_num, n_joint_types, fig_title, joint_names, axis_size, '', x_limits);
colors = lines(numel(leg_names));

for joint_idx = 1:n_joint_types
    ax = axs(joint_idx);
    hold(ax, 'on');
    for leg_idx = 1:numel(leg_names)
        data_idx = (leg_idx - 1) * n_joint_types + joint_idx;
        plot(ax, t, data_arr(:, data_idx), 'LineWidth', lw, 'Color', colors(leg_idx, :), ...
            'DisplayName', leg_names{leg_idx});
    end

    if joint_idx == 1
        lgd = legend(ax, 'show');
        lgd.Interpreter = 'none';
        lgd.Location = 'best';
        lgd.AutoUpdate = 'on';
    end
end
end

function next_fig = plotStackedGroupIfPresent(t, T, vars, prefix, fig_num, x_limits, fig_title, y_label)
cols = indexedPrefixMask(vars, prefix);
if ~any(cols)
    fprintf('[WARN] No %s* columns found. Skipping %s.\n', prefix, fig_title);
    next_fig = fig_num;
    return;
end

arr = T{:, cols};
if iscell(arr)
    arr = str2double(arr);
end
n = size(arr, 2);

axs = ensureStackedAxes(fig_num, n, fig_title, repmat({''}, 1, n), [], y_label, x_limits);
for idx = 1:n
    ax = axs(idx);
    hold(ax, 'on');
    plot(ax, t, arr(:, idx), 'LineWidth', 1.0);
end
next_fig = fig_num + 1;
end

function next_fig = plotJointGroupIfPresent(t, T, vars, prefix, fig_num, lw, axis_size, axis_labels, x_limits, fig_title)
cols = indexedPrefixMask(vars, prefix);
if ~any(cols)
    fprintf('[WARN] No %s* columns found. Skipping joint plot.\n', prefix);
    next_fig = fig_num;
    return;
end

arr = T{:, cols};
if iscell(arr)
    arr = str2double(arr);
end

joint_names = {'HAA', 'HIP', 'KNEE'};
leg_names = {'FL', 'FR', 'RL', 'RR'};
plotJointTypeSingle(fig_num, t, arr, leg_names, joint_names, lw, axis_size, x_limits, fig_title);
next_fig = fig_num + 1;
end

function axs = ensureStackedAxes(fig_num, n, fig_title, axis_labels, axis_size, group_y_label, x_limits)
fig = figure(fig_num);
appdata_key = 'QuadPlotAxes';

reuse_axes = [];
if isappdata(fig, appdata_key)
    reuse_axes = getappdata(fig, appdata_key);
end

can_reuse = ~isempty(reuse_axes) && numel(reuse_axes) == n && all(isgraphics(reuse_axes, 'axes'));
if can_reuse
    axs = reuse_axes;
    for idx = 1:n
        set(axs(idx), 'YLimMode', 'auto');
    end
    applyXAxisLimit(axs, x_limits);
    return;
end

clf(fig);
tl = tiledlayout(fig, n, 1, 'TileSpacing', 'none', 'Padding', 'none');
title(tl, fig_title, 'Interpreter', 'latex');
xlabel(tl, 'Time [s]', 'Interpreter', 'latex');
if ~isempty(group_y_label)
    ylabel(tl, y_labelText(group_y_label), 'Interpreter', 'latex');
end

axs = gobjects(1, n);
for idx = 1:n
    axs(idx) = nexttile(tl);
    grid(axs(idx), 'on');
    if idx < n
        axs(idx).XTickLabel = [];
    end

    if ~isempty(axis_labels) && idx <= numel(axis_labels) && ~isempty(axis_labels{idx})
        if isempty(axis_size)
            ylabel(axs(idx), axis_labels{idx}, 'Interpreter', 'none');
        else
            ylabel(axs(idx), axis_labels{idx}, 'FontSize', axis_size, 'Interpreter', 'none');
        end
    end
end
if n > 1
    linkaxes(axs(isgraphics(axs)), 'x');
end
applyXAxisLimit(axs, x_limits);

setappdata(fig, appdata_key, axs);
end

function applyXAxisLimit(axs, x_limits)
if isempty(axs)
    return;
end

if isempty(x_limits)
    set(axs(isgraphics(axs)), 'XLimMode', 'auto');
    return;
end

if numel(x_limits) ~= 2 || x_limits(1) >= x_limits(2)
    error('Quad_plot:InvalidXLim', ...
        'plot_x_limits must be [] or a 2-element increasing vector such as [0 1].');
end

for idx = 1:numel(axs)
    if isgraphics(axs(idx), 'axes')
        xlim(axs(idx), x_limits);
    end
end
end

function cols = indexedPrefixMask(vars, prefix)
pattern = ['^' regexptranslate('escape', char(prefix)) '\d+$'];
cols = ~cellfun('isempty', regexp(cellstr(vars), pattern, 'once'));
end

function out = y_labelText(in)
out = in;
end

function out = ternary(cond, true_val, false_val)
if cond
    out = true_val;
else
    out = false_val;
end
end
