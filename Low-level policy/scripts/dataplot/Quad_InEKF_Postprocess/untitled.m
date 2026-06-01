close all; clear; clc;

% Plot per-joint signals logged by scripts/play.py to logs/data.csv:
%   - action_*
%   - joint_angle_*
%   - joint_vel_*
%   - joint_acc_*
%   - torque_*

%% User settings
csv_path = "logs/data.csv";
default_policy_hz = 55.56;
plot_time_start_s = 0.0;
plot_time_end_s = inf;

joint_position_plot_flag = true;
joint_velocity_plot_flag = false;
joint_acceleration_plot_flag = true;
joint_acceleration_qfilter_plot_flag = false;
joint_velocity_tustin_plot_flag = false;
joint_acceleration_tustin_plot_flag = false;
joint_tau_no_comp_plot_flag = true;
joint_torque_plot_flag = false;
joint_tau_comp_plot_flag = true;
ground_reaction_force_plot_flag = true;

derivative_filter_overlay_flag = true;
joint_velocity_derivative_cutoff_hz = 1.0;
joint_acceleration_derivative_cutoff_hz = 1.0;
contact_shading_plot_flag = true;
contact_shade_color = [0.85, 0.85, 0.85];
contact_shade_alpha = 0.25;

q_filter_cutoff_hz = 10.0;
q_filter_order = 1; % 1 or 2

%% Resolve and load CSV
script_dir = fileparts(mfilename("fullpath"));
project_root = fileparts(fileparts(script_dir));
resolved_csv_path = resolve_csv_path(csv_path, project_root);
fprintf("[INFO] Using CSV: %s\n", resolved_csv_path);

T = readtable(resolved_csv_path, "VariableNamingRule", "preserve");
if height(T) == 0
    error("CSV has no rows: %s", resolved_csv_path);
end
vars = string(T.Properties.VariableNames);

if any(vars == "time_s")
    t = T{:, "time_s"};
elseif any(vars == "step")
    t = T{:, "step"} / default_policy_hz;
else
    t = (0:height(T)-1)' / default_policy_hz;
end

time_mask = (t >= plot_time_start_s) & (t <= plot_time_end_s);
if ~any(time_mask)
    error("No samples in time window [%.3f, %.3f] s.", plot_time_start_s, plot_time_end_s);
end
T = T(time_mask, :);
t = t(time_mask);
vars = string(T.Properties.VariableNames);

[action, action_names] = get_group_data(T, vars, "action_");
[joint_angle, joint_names] = get_group_data(T, vars, "joint_angle_");
[joint_vel, joint_vel_names] = get_group_data(T, vars, "joint_vel_", ["joint_vel_raw_", "joint_vel_tustin_"]);
[joint_acc, joint_acc_names] = get_group_data(T, vars, "joint_acc_", ["joint_acc_raw_", "joint_acc_tustin_"]);
[joint_vel_raw, joint_vel_raw_names] = get_group_data(T, vars, "joint_vel_raw_");
[joint_acc_raw, joint_acc_raw_names] = get_group_data(T, vars, "joint_acc_raw_");
[joint_vel_tustin, joint_vel_tustin_names] = get_group_data(T, vars, "joint_vel_tustin_");
[joint_acc_tustin, joint_acc_tustin_names] = get_group_data(T, vars, "joint_acc_tustin_");
[tau_no_comp, tau_no_comp_names] = get_group_data(T, vars, "tau_no_comp_");
[tau_comp, tau_comp_names] = get_group_data(T, vars, "tau_comp_");
[torque, torque_names] = get_group_data(T, vars, "torque_");
[contact_state, contact_names] = get_group_data(T, vars, "contact_");
[W_grf_x, W_grf_x_names] = get_group_data(T, vars, "W_grf_x_");
[W_grf_y, W_grf_y_names] = get_group_data(T, vars, "W_grf_y_");
[W_grf_z, W_grf_z_names] = get_group_data(T, vars, "W_grf_z_");

[all_joint_names, n_joint] = build_joint_name_list(action, action_names, joint_angle, joint_names, joint_vel, joint_vel_names, joint_acc, joint_acc_names, torque, torque_names);
if n_joint == 0
    error("No action_/joint_angle_/joint_vel_/joint_acc_/torque_ columns found in %s", resolved_csv_path);
end

action = align_to_joint_names(action, action_names, all_joint_names, height(T));
joint_angle = align_to_joint_names(joint_angle, joint_names, all_joint_names, height(T));
joint_vel = align_to_joint_names(joint_vel, joint_vel_names, all_joint_names, height(T));
joint_acc = align_to_joint_names(joint_acc, joint_acc_names, all_joint_names, height(T));
joint_vel_raw = align_to_joint_names(joint_vel_raw, joint_vel_raw_names, all_joint_names, height(T));
joint_acc_raw = align_to_joint_names(joint_acc_raw, joint_acc_raw_names, all_joint_names, height(T));
joint_vel_tustin = align_to_joint_names(joint_vel_tustin, joint_vel_tustin_names, all_joint_names, height(T));
joint_acc_tustin = align_to_joint_names(joint_acc_tustin, joint_acc_tustin_names, all_joint_names, height(T));
tau_no_comp = align_to_joint_names(tau_no_comp, tau_no_comp_names, all_joint_names, height(T));
tau_comp = align_to_joint_names(tau_comp, tau_comp_names, all_joint_names, height(T));
torque = align_to_joint_names(torque, torque_names, all_joint_names, height(T));

plot_order = get_leg_joint_order(all_joint_names);
all_joint_names = all_joint_names(plot_order);
action = action(:, plot_order);
joint_angle = joint_angle(:, plot_order);
joint_vel = joint_vel(:, plot_order);
joint_acc = joint_acc(:, plot_order);
joint_vel_raw = joint_vel_raw(:, plot_order);
joint_acc_raw = joint_acc_raw(:, plot_order);
joint_vel_tustin = joint_vel_tustin(:, plot_order);
joint_acc_tustin = joint_acc_tustin(:, plot_order);
tau_no_comp = tau_no_comp(:, plot_order);
tau_comp = tau_comp(:, plot_order);
torque = torque(:, plot_order);

joint_vel_df = [];
joint_acc_df = [];
if derivative_filter_overlay_flag
    derivative_filter_dt = infer_sample_dt(t, 1 / default_policy_hz);
    fprintf("[INFO] Tustin derivative dt = %.6f s (vel cutoff = %.1f Hz, acc cutoff = %.1f Hz)\n", ...
        derivative_filter_dt, joint_velocity_derivative_cutoff_hz, joint_acceleration_derivative_cutoff_hz);
    joint_vel_df = apply_tustin_derivative_filter(joint_angle, t, joint_velocity_derivative_cutoff_hz);
    joint_acc_df = apply_tustin_derivative_filter(joint_vel_df, t, joint_acceleration_derivative_cutoff_hz);
end

joint_acc_qf = [];
if joint_acceleration_qfilter_plot_flag
    joint_acc_qf = apply_tustin_q_filter(joint_acc, t, q_filter_cutoff_hz, q_filter_order);
end

plot_velocity_tracking(t, T, vars);
if joint_position_plot_flag
    plot_joint_action_overlay(t, action, joint_angle, all_joint_names, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_velocity_plot_flag
    plot_joint_velocity(t, joint_vel, joint_vel_df, all_joint_names, joint_velocity_derivative_cutoff_hz, derivative_filter_overlay_flag, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_acceleration_plot_flag
    plot_joint_acceleration(t, joint_acc, joint_acc_df, all_joint_names, joint_acceleration_derivative_cutoff_hz, derivative_filter_overlay_flag, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_acceleration_qfilter_plot_flag
    plot_joint_acceleration_qfilter(t, joint_acc_qf, all_joint_names, q_filter_cutoff_hz, q_filter_order, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_velocity_tustin_plot_flag
    plot_joint_velocity_tustin(t, joint_vel_tustin, all_joint_names, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_acceleration_tustin_plot_flag
    plot_joint_acceleration_tustin(t, joint_acc_tustin, all_joint_names, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_tau_no_comp_plot_flag
    plot_joint_tau_no_comp(t, tau_no_comp, all_joint_names, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_torque_plot_flag
    plot_joint_torque(t, torque, all_joint_names, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if joint_tau_comp_plot_flag
    plot_joint_tau_comp(t, tau_comp, all_joint_names, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end
if ground_reaction_force_plot_flag
    plot_ground_reaction_force(t, W_grf_x, W_grf_x_names, W_grf_y, W_grf_y_names, W_grf_z, W_grf_z_names, contact_state, contact_names, contact_shading_plot_flag, contact_shade_color, contact_shade_alpha);
end

fprintf("[DONE] Plot complete from: %s\n", resolved_csv_path);

%% Local functions
function resolved_csv_path = resolve_csv_path(csv_path, project_root)
    if isfile(csv_path)
        resolved_csv_path = csv_path;
        return;
    end

    candidate = fullfile(project_root, csv_path);
    if isfile(candidate)
        resolved_csv_path = string(candidate);
        return;
    end

    error("CSV not found: %s", csv_path);
end

function [arr, names] = get_group_data(T, vars, prefix, exclude_prefixes)
    if nargin < 4
        exclude_prefixes = strings(1, 0);
    else
        exclude_prefixes = string(exclude_prefixes);
    end
    cols = startsWith(vars, prefix);
    for i = 1:numel(exclude_prefixes)
        cols = cols & ~startsWith(vars, exclude_prefixes(i));
    end
    arr = [];
    names = strings(1, 0);
    if any(cols)
        arr = T{:, cols};
        if iscell(arr), arr = str2double(arr); end
        names = erase(vars(cols), prefix);
    end
end

function [all_names, n_joint] = build_joint_name_list(action, action_names, joint_angle, joint_names, joint_vel, joint_vel_names, joint_acc, joint_acc_names, torque, torque_names)
    all_names = strings(1, 0);
    candidates = [joint_names, joint_vel_names, joint_acc_names, torque_names, action_names];
    for i = 1:numel(candidates)
        name_i = candidates(i);
        if strlength(name_i) == 0
            continue;
        end
        if ~any(all_names == name_i)
            all_names(end+1) = name_i; %#ok<AGROW>
        end
    end

    n_joint = max([size(action, 2), size(joint_angle, 2), size(joint_vel, 2), size(joint_acc, 2), size(torque, 2)]);
    if n_joint == 0
        return;
    end
    if isempty(all_names)
        all_names = "joint_" + string(1:n_joint);
    elseif numel(all_names) < n_joint
        all_names = [all_names, "joint_" + string(numel(all_names)+1:n_joint)];
    else
        all_names = all_names(1:n_joint);
    end
end

function [all_names, n_signal] = build_signal_name_list(varargin)
    all_names = strings(1, 0);
    n_signal = 0;
    for i = 1:nargin
        names_i = string(varargin{i});
        if isempty(names_i)
            continue;
        end
        n_signal = max(n_signal, numel(names_i));
        for j = 1:numel(names_i)
            name_j = names_i(j);
            if strlength(name_j) == 0
                continue;
            end
            if ~any(all_names == name_j)
                all_names(end+1) = name_j; %#ok<AGROW>
            end
        end
    end
    if n_signal == 0
        return;
    end
    if isempty(all_names)
        all_names = "signal_" + string(1:n_signal);
    elseif numel(all_names) < n_signal
        all_names = [all_names, "signal_" + string(numel(all_names)+1:n_signal)];
    else
        all_names = all_names(1:n_signal);
    end
end

function y = apply_tustin_q_filter(x, t, cutoff_hz, filter_order)
    y = x;
    if isempty(x) || size(x, 1) < 2
        return;
    end
    if nargin < 4
        filter_order = 1;
    end
    filter_order = round(double(filter_order));
    filter_order = max(1, min(2, filter_order));
    dt = infer_sample_dt(t, 1 / 50);

    wc = 2 * pi * cutoff_hz;
    K = 2 / dt;
    denom = K + wc;
    b0 = wc / denom;
    b1 = wc / denom;
    a1 = (wc - K) / denom;

    for i = 1:size(x, 2)
        x_col = x(:, i);
        valid = isfinite(x_col);
        if ~any(valid)
            y(:, i) = nan(size(x_col));
            continue;
        end
        if all(valid)
            x_filled = x_col;
        else
            idx = (1:numel(x_col))';
            if nnz(valid) == 1
                x_filled = repmat(x_col(valid), size(x_col));
            else
                x_filled = x_col;
                x_filled(~valid) = interp1(idx(valid), x_col(valid), idx(~valid), "linear", "extrap");
            end
        end
        y_col = x_filled;
        for k = 1:filter_order
            y_col = filter([b0, b1], [1, a1], y_col);
        end
        y(:, i) = y_col;
    end
end

function y = apply_tustin_derivative_filter(x, t, cutoff_hz)
    y = x;
    if isempty(x) || size(x, 1) < 2 || cutoff_hz <= 0
        return;
    end

    dt = infer_sample_dt(t, 1 / 50);

    tau = 1 / (2 * pi * cutoff_hz);
    K = 2 / dt;
    denom = tau * K + 1;
    b0 = K / denom;
    b1 = -K / denom;
    a1 = (1 - tau * K) / denom;

    y = nan(size(x));
    for i = 1:size(x, 2)
        x_col = x(:, i);
        valid = isfinite(x_col);
        if ~any(valid)
            continue;
        end

        if all(valid)
            x_filled = x_col;
        else
            idx = (1:numel(x_col))';
            if nnz(valid) == 1
                x_filled = repmat(x_col(valid), size(x_col));
            else
                x_filled = x_col;
                x_filled(~valid) = interp1(idx(valid), x_col(valid), idx(~valid), "linear", "extrap");
            end
        end

        y_col = filter([b0, b1], [1, a1], x_filled);
        y_col(~valid) = nan;
        y(:, i) = y_col;
    end
end

function dt = infer_sample_dt(t, default_dt)
    dt_vec = diff(t);
    dt_vec = dt_vec(isfinite(dt_vec) & (dt_vec > 0));
    if isempty(dt_vec)
        dt = default_dt;
    else
        dt = median(dt_vec);
    end
end

function out = align_to_joint_names(arr, names, all_names, n_rows)
    out = nan(n_rows, numel(all_names));
    if isempty(arr)
        return;
    end
    if isempty(names)
        n_copy = min(size(arr, 2), numel(all_names));
        out(:, 1:n_copy) = arr(:, 1:n_copy);
        return;
    end

    for i = 1:size(arr, 2)
        idx = find(all_names == names(i), 1);
        if ~isempty(idx)
            out(:, idx) = arr(:, i);
        elseif i <= numel(all_names) && all(isnan(out(:, i)))
            out(:, i) = arr(:, i);
        end
    end
end

function order = get_leg_joint_order(joint_names)
    n = numel(joint_names);
    key = zeros(n, 3);
    for i = 1:n
        token = upper(regexprep(joint_names(i), "[^A-Za-z0-9]", ""));
        leg_rank = get_leg_rank(token);
        joint_rank = get_joint_rank(token);
        key(i, :) = [leg_rank, joint_rank, i];
    end
    [~, idx] = sortrows(key, [1, 2, 3]);
    order = idx(:)';
end

function rank = get_leg_rank(token)
    if startsWith(token, "FL")
        rank = 1;
    elseif startsWith(token, "FR")
        rank = 2;
    elseif startsWith(token, "RL") || startsWith(token, "HL")
        rank = 3;
    elseif startsWith(token, "RR") || startsWith(token, "HR")
        rank = 4;
    else
        rank = 99;
    end
end

function rank = get_joint_rank(token)
    if contains(token, "HAA")
        rank = 1;
    elseif contains(token, "HIP") || contains(token, "THIGH")
        rank = 2;
    elseif contains(token, "KNEE") || contains(token, "CALF")
        rank = 3;
    else
        rank = 99;
    end
end

function [n_rows, n_cols] = get_tile_shape(n)
    if n == 12
        n_rows = 4;
        n_cols = 3;
    else
        n_cols = 3;
        n_rows = ceil(n / n_cols);
    end
end

function axs = get_or_create_tiled_axes(fig_tag, fig_name, layout_title, n_rows, n_cols, n_tiles, tile_spacing, padding)
    fig_tag = char(fig_tag);
    fig_name = char(fig_name);
    layout_title = char(layout_title);
    if nargin < 7
        tile_spacing = "none";
    end
    if nargin < 8
        padding = "none";
    end

    fig = findobj(0, "Type", "figure", "Tag", fig_tag);
    if isempty(fig)
        fig = figure("Name", fig_name, "Tag", fig_tag);
    else
        fig = fig(1);
        figure(fig);
    end

    reuse = false;
    if isstruct(fig.UserData) && isfield(fig.UserData, "tile_axes")
        maybe_axes = fig.UserData.tile_axes;
        if numel(maybe_axes) == n_tiles && all(isgraphics(maybe_axes))
            axs = maybe_axes(:)';
            reuse = true;
        end
    end

    if ~reuse
        clf(fig);
        tl = tiledlayout(fig, n_rows, n_cols, "TileSpacing", tile_spacing, "Padding", padding);
        title(tl, layout_title);
        axs = gobjects(1, n_tiles);
        for i = 1:n_tiles
            axs(i) = nexttile(tl, i);
        end
        user_data = fig.UserData;
        if ~isstruct(user_data)
            user_data = struct();
        end
        user_data.tile_axes = axs;
        fig.UserData = user_data;
    end

    for i = 1:numel(axs)
        hold(axs(i), "on");
        grid(axs(i), "on");
    end
end

function [group_indices, group_labels] = get_joint_type_groups(joint_names)
    names = upper(string(joint_names));
    group_labels = ["HAA", "HIP", "KNEE"];
    group_indices = {
        find(contains(names, "HAA")), ...
        find(contains(names, "HIP") | contains(names, "THIGH")), ...
        find(contains(names, "KNEE") | contains(names, "CALF"))
    };
end

function plot_joint_action_overlay(t, action, joint_angle, joint_names, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0
        fprintf("[WARN] No joint columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        n_rows = 4;
        n_cols = 1;
        tag_suffix = lower(char(group_labels(g)));
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_angle_action_" + tag_suffix, ...
            "Joint Angle And Action - " + group_labels(g), ...
            "Joint Angle / Action - " + group_labels(g), ...
            n_rows, n_cols, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h1 = plot(ax, t, joint_angle(:, idx(i)), "LineWidth", 0.9, "Color", [0.00, 0.45, 0.74]);
            h2 = plot(ax, t, action(:, idx(i)), "LineWidth", 0.9, "Color", [0.85, 0.33, 0.10]);
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "angle/action");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                lg = legend(ax, [h1, h2], {"joint angle", "action"}, "Location", "best");
                lg.AutoUpdate = "off";
            end
        end
        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_torque(t, torque, joint_names, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0
        fprintf("[WARN] No joint columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        n_rows = 4;
        n_cols = 1;
        tag_suffix = lower(char(group_labels(g)));
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_torque_" + tag_suffix, ...
            "Joint Torque - " + group_labels(g), ...
            "Joint Torque - " + group_labels(g), ...
            n_rows, n_cols, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h = plot(ax, t, torque(:, idx(i)), "LineWidth", 0.9, "Color", [0.47, 0.67, 0.19]);
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "torque [Nm]");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                lg = legend(ax, h, {"torque"}, "Location", "best");
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_tau_no_comp(t, tau_no_comp, joint_names, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0
        fprintf("[WARN] No joint columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        n_rows = 4;
        n_cols = 1;
        tag_suffix = lower(char(group_labels(g)));
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_tau_no_comp_" + tag_suffix, ...
            "Joint Tau No Comp - " + group_labels(g), ...
            "Joint Tau No Comp - " + group_labels(g), ...
            n_rows, n_cols, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h = plot(ax, t, tau_no_comp(:, idx(i)), "LineWidth", 0.9, "Color", [0.13, 0.55, 0.13]);
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "tau\_no\_comp [Nm]");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                lg = legend(ax, h, {"tau\_no\_comp"}, "Location", "best");
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_tau_comp(t, tau_comp, joint_names, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0
        fprintf("[WARN] No joint columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        n_rows = 4;
        n_cols = 1;
        tag_suffix = lower(char(group_labels(g)));
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_tau_comp_" + tag_suffix, ...
            "Joint Tau Comp - " + group_labels(g), ...
            "Joint Tau Comp - " + group_labels(g), ...
            n_rows, n_cols, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h = plot(ax, t, tau_comp(:, idx(i)), "LineWidth", 0.9, "Color", [0.49, 0.18, 0.56]);
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "tau\_comp [Nm]");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                lg = legend(ax, h, {"tau\_comp"}, "Location", "best");
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_velocity(t, joint_vel, joint_vel_df, joint_names, cutoff_hz, overlay_flag, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0
        fprintf("[WARN] No joint columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        n_rows = 4;
        n_cols = 1;
        tag_suffix = lower(char(group_labels(g)));
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_velocity_" + tag_suffix, ...
            "Joint Velocity - " + group_labels(g), ...
            "Joint Velocity - " + group_labels(g), ...
            n_rows, n_cols, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h_raw = plot(ax, t, joint_vel(:, idx(i)), "LineWidth", 0.9, "Color", [0.30, 0.75, 0.93]);
            h_est = [];
            if overlay_flag && ~isempty(joint_vel_df) && size(joint_vel_df, 2) >= idx(i)
                h_est = plot(ax, t, joint_vel_df(:, idx(i)), "LineWidth", 0.9, "Color", [0.85, 0.33, 0.10]);
            end
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "vel [rad/s]");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                if ~isempty(h_est)
                    lg = legend(ax, [h_raw, h_est], {"joint vel raw", sprintf("joint vel est (s/(tau s+1), %.1f Hz)", cutoff_hz)}, "Location", "best");
                else
                    lg = legend(ax, h_raw, {"joint vel raw"}, "Location", "best");
                end
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_acceleration(t, joint_acc, joint_acc_df, joint_names, cutoff_hz, overlay_flag, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0
        fprintf("[WARN] No joint columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        n_rows = 4;
        n_cols = 1;
        tag_suffix = lower(char(group_labels(g)));
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_acceleration_" + tag_suffix, ...
            "Joint Acceleration - " + group_labels(g), ...
            "Joint Acceleration - " + group_labels(g), ...
            n_rows, n_cols, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h_raw = plot(ax, t, joint_acc(:, idx(i)), "LineWidth", 0.9, "Color", [0.49, 0.18, 0.56]);
            h_est = [];
            if overlay_flag && ~isempty(joint_acc_df) && size(joint_acc_df, 2) >= idx(i)
                h_est = plot(ax, t, joint_acc_df(:, idx(i)), "LineWidth", 0.9, "Color", [0.13, 0.55, 0.13]);
            end
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "acc [rad/s^2]");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                if ~isempty(h_est)
                    lg = legend(ax, [h_raw, h_est], {"joint acc raw", sprintf("joint acc est (s/(tau s+1), %.1f Hz)", cutoff_hz)}, "Location", "best");
                else
                    lg = legend(ax, h_raw, {"joint acc raw"}, "Location", "best");
                end
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_acceleration_qfilter(t, joint_acc_qf, joint_names, cutoff_hz, filter_order, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0
        fprintf("[WARN] No joint columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        n_rows = 4;
        n_cols = 1;
        tag_suffix = lower(char(group_labels(g)));
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_acc_qfilter_" + tag_suffix, ...
            "Joint Acceleration Q-Filter - " + group_labels(g), ...
            sprintf("Joint Acceleration Q-Filter (Tustin %d-order, %.1f Hz) - %s", filter_order, cutoff_hz, char(group_labels(g))), ...
            n_rows, n_cols, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h = plot(ax, t, joint_acc_qf(:, idx(i)), "LineWidth", 0.9, "Color", [0.13, 0.55, 0.13]);
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "acc_qf [rad/s^2]", "Interpreter", "none");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                lg = legend(ax, h, {"joint acc qf"}, "Location", "best");
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_velocity_tustin(t, joint_vel_tustin, joint_names, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0 || isempty(joint_vel_tustin)
        fprintf("[WARN] No joint_vel_tustin columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_velocity_tustin_" + lower(char(group_labels(g))), ...
            "Joint Velocity Tustin - " + group_labels(g), ...
            "Joint Velocity Tustin - " + group_labels(g), ...
            4, 1, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h = plot(ax, t, joint_vel_tustin(:, idx(i)), "LineWidth", 0.9, "Color", [0.85, 0.33, 0.10]);
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "vel_tustin [rad/s]", "Interpreter", "none");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                lg = legend(ax, h, {"joint vel tustin"}, "Location", "best");
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function plot_joint_acceleration_tustin(t, joint_acc_tustin, joint_names, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    if numel(joint_names) == 0 || isempty(joint_acc_tustin)
        fprintf("[WARN] No joint_acc_tustin columns to plot.\n");
        return;
    end

    [group_indices, group_labels] = get_joint_type_groups(joint_names);
    for g = 1:numel(group_labels)
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end
        n_group = numel(idx);
        axs = get_or_create_tiled_axes( ...
            "dataplot_joint_acceleration_tustin_" + lower(char(group_labels(g))), ...
            "Joint Acceleration Tustin - " + group_labels(g), ...
            "Joint Acceleration Tustin - " + group_labels(g), ...
            4, 1, n_group, "compact", "compact");

        for i = 1:n_group
            ax = axs(i);
            h = plot(ax, t, joint_acc_tustin(:, idx(i)), "LineWidth", 0.9, "Color", [0.13, 0.55, 0.13]);
            if shade_contact
                overlay_contact_regions(ax, t, get_contact_signal_for_joint(joint_names(idx(i)), contact_state, contact_names), shade_color, shade_alpha);
            end
            ylabel(ax, "acc_tustin [rad/s^2]", "Interpreter", "none");
            title(ax, get_joint_label(joint_names, idx(i)), "Interpreter", "none");
            if i == n_group
                xlabel(ax, "t [s]");
            else
                ax.XTickLabel = [];
            end
            if i == 1
                lg = legend(ax, h, {"joint acc tustin"}, "Location", "best");
                lg.AutoUpdate = "off";
            end
        end

        if numel(axs) > 1
            linkaxes(axs, "x");
        end
    end
end

function label = get_joint_label(joint_names, idx)
    if idx <= numel(joint_names)
        label = joint_names(idx);
    else
        label = "joint_" + string(idx);
    end
end

function label = get_signal_label(signal_names, idx, fallback_prefix)
    if idx <= numel(signal_names)
        label = signal_names(idx);
    else
        label = fallback_prefix + "_" + string(idx);
    end
end

function contact_signal = get_contact_signal_for_joint(joint_name, contact_state, contact_names)
    contact_signal = [];
    if isempty(contact_state) || isempty(contact_names)
        return;
    end
    leg_token = get_leg_token_from_name(joint_name);
    if strlength(leg_token) == 0
        return;
    end
    normalized_contacts = upper(regexprep(string(contact_names), "[^A-Za-z0-9]", ""));
    match_idx = find(startsWith(normalized_contacts, leg_token) & contains(normalized_contacts, "FOOT"), 1);
    if isempty(match_idx)
        return;
    end
    contact_signal = contact_state(:, match_idx);
end

function leg_token = get_leg_token_from_name(name)
    token = upper(regexprep(string(name), "[^A-Za-z0-9]", ""));
    if startsWith(token, "FL")
        leg_token = "FL";
    elseif startsWith(token, "FR")
        leg_token = "FR";
    elseif startsWith(token, "RL") || startsWith(token, "HL")
        leg_token = "RL";
    elseif startsWith(token, "RR") || startsWith(token, "HR")
        leg_token = "RR";
    else
        leg_token = "";
    end
end

function overlay_contact_regions(ax, t, contact_signal, shade_color, shade_alpha)
    if isempty(contact_signal)
        return;
    end
    contact_mask = isfinite(contact_signal) & (contact_signal > 0.5);
    if ~any(contact_mask)
        return;
    end

    y_limits = ylim(ax);
    dt_vec = diff(t);
    dt_vec = dt_vec(isfinite(dt_vec) & (dt_vec > 0));
    if isempty(dt_vec)
        dt_default = 0.0;
    else
        dt_default = median(dt_vec);
    end

    edge_mask = diff([false; contact_mask(:); false]);
    start_idx = find(edge_mask == 1);
    stop_idx = find(edge_mask == -1) - 1;

    for i = 1:numel(start_idx)
        x0 = t(start_idx(i));
        x1 = t(stop_idx(i));
        if x1 <= x0
            x1 = x0 + dt_default;
        elseif stop_idx(i) < numel(t)
            x1 = t(stop_idx(i) + 1);
        else
            x1 = x1 + dt_default;
        end
        patch(ax, [x0, x1, x1, x0], [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], shade_color, ...
            "FaceAlpha", shade_alpha, "EdgeColor", "none", "HandleVisibility", "off");
    end

    ylim(ax, y_limits);
    for child = transpose(ax.Children)
        if isa(child, "matlab.graphics.primitive.Patch")
            uistack(child, "bottom");
        end
    end
end

function plot_velocity_tracking(t, T, vars)
    required = ["cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel_z", ...
                "base_lin_vel_x", "base_lin_vel_y", "base_ang_vel_z"];
    if ~all(ismember(required, vars))
        fprintf("[WARN] Command/velocity tracking columns not found. Skip tracking plot.\n");
        return;
    end

    cmd_vx = T{:, "cmd_lin_vel_x"};
    cmd_vy = T{:, "cmd_lin_vel_y"};
    cmd_wz = T{:, "cmd_ang_vel_z"};
    act_vx = T{:, "base_lin_vel_x"};
    act_vy = T{:, "base_lin_vel_y"};
    act_wz = T{:, "base_ang_vel_z"};

    rmse_vx = sqrt(mean((act_vx - cmd_vx).^2, "omitnan"));
    rmse_vy = sqrt(mean((act_vy - cmd_vy).^2, "omitnan"));
    rmse_wz = sqrt(mean((act_wz - cmd_wz).^2, "omitnan"));
    axs = get_or_create_tiled_axes( ...
        "dataplot_velocity_tracking", ...
        "Command Tracking", ...
        "Velocity Command Tracking", ...
        3, 1, 3, "compact", "compact");

    ax1 = axs(1);
    h11 = plot(ax1, t, cmd_vx, "LineWidth", 1.0, "Color", [0.85, 0.33, 0.10]);
    h12 = plot(ax1, t, act_vx, "LineWidth", 1.0, "Color", [0.00, 0.45, 0.74]);
    ylabel(ax1, "v_x [m/s]");
    title(ax1, sprintf("x tracking (RMSE=%.4f)", rmse_vx));
    lg = legend(ax1, [h11, h12], {"command", "actual"}, "Location", "best");
    lg.AutoUpdate = "off";

    ax2 = axs(2);
    plot(ax2, t, cmd_vy, "LineWidth", 1.0, "Color", [0.85, 0.33, 0.10]);
    plot(ax2, t, act_vy, "LineWidth", 1.0, "Color", [0.00, 0.45, 0.74]);
    ylabel(ax2, "v_y [m/s]");
    title(ax2, sprintf("y tracking (RMSE=%.4f)", rmse_vy));
    ax2.XTickLabel = [];

    ax3 = axs(3);
    plot(ax3, t, cmd_wz, "LineWidth", 1.0, "Color", [0.85, 0.33, 0.10]);
    plot(ax3, t, act_wz, "LineWidth", 1.0, "Color", [0.00, 0.45, 0.74]);
    ylabel(ax3, "\omega_z [rad/s]");
    xlabel(ax3, "t [s]");
    title(ax3, sprintf("yaw tracking (RMSE=%.4f)", rmse_wz));

    linkaxes(axs, "x");
end

function plot_ground_reaction_force(t, W_grf_x, W_grf_x_names, W_grf_y, W_grf_y_names, W_grf_z, W_grf_z_names, contact_state, contact_names, shade_contact, shade_color, shade_alpha)
    [foot_names, n_foot] = build_signal_name_list(W_grf_x_names, W_grf_y_names, W_grf_z_names);
    if n_foot == 0
        fprintf("[WARN] No W_grf_x_/W_grf_y_/W_grf_z_ columns found. Skip GRF plot.\n");
        return;
    end

    W_grf_x = align_to_joint_names(W_grf_x, W_grf_x_names, foot_names, numel(t));
    W_grf_y = align_to_joint_names(W_grf_y, W_grf_y_names, foot_names, numel(t));
    W_grf_z = align_to_joint_names(W_grf_z, W_grf_z_names, foot_names, numel(t));
    W_grf_norm = sqrt(W_grf_x.^2 + W_grf_y.^2 + W_grf_z.^2);

    plot_order = get_leg_joint_order(foot_names);
    foot_names = foot_names(plot_order);
    W_grf_x = W_grf_x(:, plot_order);
    W_grf_y = W_grf_y(:, plot_order);
    W_grf_z = W_grf_z(:, plot_order);
    W_grf_norm = W_grf_norm(:, plot_order);

    [n_rows, n_cols] = get_tile_shape(numel(foot_names));
    axs = get_or_create_tiled_axes( ...
        "dataplot_ground_reaction_force", ...
        "Ground Reaction Force", ...
        "Ground Reaction Force Per Foot", ...
        n_rows, n_cols, numel(foot_names), "compact", "compact");

    for i = 1:numel(foot_names)
        ax = axs(i);
        hx = plot(ax, t, W_grf_x(:, i), "LineWidth", 0.9, "Color", [0.85, 0.33, 0.10]);
        hy = plot(ax, t, W_grf_y(:, i), "LineWidth", 0.9, "Color", [0.00, 0.45, 0.74]);
        hz = plot(ax, t, W_grf_z(:, i), "LineWidth", 0.9, "Color", [0.47, 0.67, 0.19]);
        hn = plot(ax, t, W_grf_norm(:, i), "--", "LineWidth", 0.9, "Color", [0.49, 0.18, 0.56]);
        if shade_contact
            overlay_contact_regions(ax, t, get_contact_signal_for_joint(foot_names(i), contact_state, contact_names), shade_color, shade_alpha);
        end
        ylabel(ax, "force [N]");
        title(ax, get_signal_label(foot_names, i, "foot"), "Interpreter", "none");
        if i > (numel(foot_names) - n_cols)
            xlabel(ax, "t [s]");
        else
            ax.XTickLabel = [];
        end
        if i == 1
            lg = legend(ax, [hx, hy, hz, hn], {"W\_grf x", "W\_grf y", "W\_grf z", "|W\_grf|"}, "Location", "best");
            lg.AutoUpdate = "off";
        end
    end

    if numel(axs) > 1
        linkaxes(axs, "x");
    end
end

setFigurePositions(8);
