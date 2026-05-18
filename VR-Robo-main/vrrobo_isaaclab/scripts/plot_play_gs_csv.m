% plot_play_gs_csv.m
% Plot the single CSV diagnostics saved automatically by scripts/rsl_rl/play_gs.py.
%
% Usage from MATLAB:
%   run('/home/teamquad/Desktop/JHKim/[3] Vision_RL/VR-Robo-main/vrrobo_isaaclab/scripts/plot_play_gs_csv.m')
%
% Change only this path when plotting another play log.

clc;
clear;
close all;

CSV_PATH = '/home/teamquad/Desktop/JHKim/[3] Vision_RL/VR-Robo-main/vrrobo_isaaclab/logs/rsl_rl/mclquad_flat_cone_camera/2026-05-13_13-29-06/csv_logs/play_model_400.csv';

if ~isfile(CSV_PATH)
    error('CSV file not found: %s', CSV_PATH);
end

[csv_dir, csv_name, ~] = fileparts(CSV_PATH);

T = readtable(CSV_PATH);
fprintf('[INFO] Loaded CSV: %s\n', CSV_PATH);

t = get_col(T, 't_policy_s');
if isempty(t)
    t = (0:height(T)-1)';
end
done = get_col(T, 'done');

cmd = cols(T, {'cmd_vx_body', 'cmd_vy_body', 'cmd_yaw_rate'});
root_v_b = cols(T, {'root_lin_vel_b_x', 'root_lin_vel_b_y', 'root_lin_vel_b_z'});
root_w_b = cols(T, {'root_ang_vel_b_x', 'root_ang_vel_b_y', 'root_ang_vel_b_z'});
root_pos_w = cols(T, {'root_pos_w_x', 'root_pos_w_y', 'root_pos_w_z'});
root_quat_w = cols(T, {'root_quat_w_w', 'root_quat_w_x', 'root_quat_w_y', 'root_quat_w_z'});
cone_w = cols(T, {'cone_red_pos_w_x', 'cone_red_pos_w_y', 'cone_red_pos_w_z'});
reward = get_col(T, 'reward');
action = prefix_cols(T, 'action_');
q = prefix_cols(T, 'joint_pos_', {'joint_pos_target_'});
dq = prefix_cols(T, 'joint_vel_');
q_des = prefix_cols(T, 'joint_pos_target_');

rpy = [];
if size(root_quat_w, 2) >= 4
    rpy = quat_wxyz_to_rpy(root_quat_w);
end

% -------------------------------------------------------------------------
% Figure 1: command tracking
% -------------------------------------------------------------------------
if size(cmd, 2) >= 3 && size(root_v_b, 2) >= 2 && size(root_w_b, 2) >= 3
    figure('Name', 'CSV Command Tracking', 'NumberTitle', 'off', 'Position', [60 60 1250 520]);
    labels = {'v_x body [m/s]', 'v_y body [m/s]', 'yaw rate [rad/s]'};
    actual = {root_v_b(:,1), root_v_b(:,2), root_w_b(:,3)};
    for k = 1:3
        subplot(1, 3, k); hold on;
        plot(t, cmd(:,k), '--', 'LineWidth', 1.5, 'DisplayName', 'command');
        plot(t, actual{k}, 'LineWidth', 1.2, 'DisplayName', 'actual');
        mark_done_lines(t, done);
        xlabel('time [s]'); ylabel(labels{k}); title(labels{k});
        legend('Location', 'best'); grid on;
    end
    sgtitle('High-level command vs simulated body velocity');
end

% -------------------------------------------------------------------------
% Figure 2: policy raw output and scaled command
% -------------------------------------------------------------------------
if ~isempty(action) || size(cmd, 2) >= 3
    figure('Name', 'CSV Policy Output and Command', 'NumberTitle', 'off', 'Position', [80 80 1250 620]);
    names = {'x', 'y', 'yaw'};
    for k = 1:3
        subplot(2, 3, k); hold on;
        if size(action, 2) >= k
            plot(t, action(:,k), 'LineWidth', 1.1);
        end
        yline(0, 'k:'); yline(1, 'k:'); yline(-1, 'k:');
        xlabel('time [s]'); ylabel('raw action'); title(['raw action ', names{k}]); grid on;

        subplot(2, 3, 3+k); hold on;
        if size(cmd, 2) >= k
            plot(t, cmd(:,k), 'LineWidth', 1.1);
        end
        yline(0, 'k:');
        xlabel('time [s]'); ylabel('command'); title(['command ', names{k}]); grid on;
    end
    sgtitle('Policy action and tanh(action) * velocity\_range');
end

% -------------------------------------------------------------------------
% Figure 3: XY trajectory and target cone
% -------------------------------------------------------------------------
if size(root_pos_w, 2) >= 2
    figure('Name', 'CSV Trajectory', 'NumberTitle', 'off', 'Position', [120 120 780 700]);
    plot(root_pos_w(:,1), root_pos_w(:,2), 'LineWidth', 1.4, 'DisplayName', 'robot root'); hold on;
    if size(rpy, 2) >= 3
        arrow_stride = max(1, floor(size(root_pos_w, 1) / 20));
        arrow_idx = 1:arrow_stride:size(root_pos_w, 1);
        arrow_len = 0.18;
        quiver( ...
            root_pos_w(arrow_idx,1), root_pos_w(arrow_idx,2), ...
            arrow_len * cos(rpy(arrow_idx,3)), arrow_len * sin(rpy(arrow_idx,3)), ...
            0, 'Color', [0.05 0.35 0.95], 'LineWidth', 1.1, 'MaxHeadSize', 1.8, ...
            'DisplayName', 'robot yaw');
    end
    if size(cone_w, 2) >= 2
        plot(cone_w(:,1), cone_w(:,2), 'r--', 'LineWidth', 1.0, 'DisplayName', 'cone');
        scatter(cone_w(1,1), cone_w(1,2), 80, 'r', 'filled', 'DisplayName', 'cone start');
    end
    scatter(root_pos_w(1,1), root_pos_w(1,2), 60, 'g', 'filled', 'DisplayName', 'start');
    scatter(root_pos_w(end,1), root_pos_w(end,2), 60, 'k', 'filled', 'DisplayName', 'end');
    axis equal; grid on; xlabel('x [m]'); ylabel('y [m]');
    title('Robot trajectory');
    legend('Location', 'best');
end

% -------------------------------------------------------------------------
% Figure 4: distance, reward, done
% -------------------------------------------------------------------------
figure('Name', 'CSV Task Signals', 'NumberTitle', 'off', 'Position', [140 140 1050 720]);
subplot(3, 1, 1); hold on;
if size(root_pos_w, 2) >= 3 && size(cone_w, 2) >= 3
    dist_xy = vecnorm(root_pos_w(:,1:2) - cone_w(:,1:2), 2, 2);
    dist_3d = vecnorm(root_pos_w(:,1:3) - cone_w(:,1:3), 2, 2);
    plot(t, dist_xy, 'LineWidth', 1.2, 'DisplayName', 'xy distance');
    plot(t, dist_3d, 'LineWidth', 1.2, 'DisplayName', '3d distance');
    yline(0.35, 'k--', 'reach threshold');
    legend('Location', 'best');
end
mark_done_lines(t, done); grid on; xlabel('time [s]'); ylabel('distance [m]'); title('Distance to cone');

subplot(3, 1, 2); hold on;
if ~isempty(reward), plot(t, reward, 'LineWidth', 1.1); end
mark_done_lines(t, done); grid on; xlabel('time [s]'); ylabel('reward'); title('Reward');

subplot(3, 1, 3); hold on;
if ~isempty(done), stairs(t, done, 'LineWidth', 1.1); end
grid on; xlabel('time [s]'); ylabel('done'); title('Done flags');

% -------------------------------------------------------------------------
% Figure 5: base state
% -------------------------------------------------------------------------
if size(root_pos_w, 2) >= 3 || size(root_quat_w, 2) >= 4
    xyz_names = {'x', 'y', 'z'};
    rpy_names = {'roll', 'pitch', 'yaw'};
    figure('Name', 'CSV Base State', 'NumberTitle', 'off', 'Position', [160 160 1250 800]);
    for k = 1:3
        subplot(3, 3, k); hold on;
        if size(root_pos_w, 2) >= k, plot(t, root_pos_w(:,k), 'LineWidth', 1.0); end
        grid on; xlabel('time [s]'); ylabel('m'); title(sprintf('root pos %s', xyz_names{k}));

        subplot(3, 3, 3+k); hold on;
        if size(rpy, 2) >= k, plot(t, rpy(:,k), 'LineWidth', 1.0); end
        grid on; xlabel('time [s]'); ylabel('rad'); title(sprintf('base %s', rpy_names{k}));

        subplot(3, 3, 6+k); hold on;
        if size(root_v_b, 2) >= k, plot(t, root_v_b(:,k), 'LineWidth', 1.0); end
        grid on; xlabel('time [s]'); ylabel('m/s'); title(sprintf('body linear vel %s', xyz_names{k}));
    end
    sgtitle('Base pose and velocity');
end

% -------------------------------------------------------------------------
% Figure 6-7: policy-step joint signals
% -------------------------------------------------------------------------
if ~isempty(q)
    plot_joint_grid(t, q, q_des, 'Policy-step Joint Positions', 'rad', 'actual', 'target');
end
if ~isempty(dq)
    plot_joint_grid(t, dq, [], 'Policy-step Joint Velocities', 'rad/s', 'actual', '');
end

fprintf('[DONE] Generated plots from CSV: %s\n', fullfile(csv_dir, csv_name));

% =========================================================================
% Local helper functions
% =========================================================================
function y = get_col(T, name)
    if any(strcmp(T.Properties.VariableNames, name))
        y = double(T.(name));
    else
        y = [];
    end
end

function y = cols(T, names)
    y = [];
    for k = 1:numel(names)
        col = get_col(T, names{k});
        if isempty(col)
            return;
        end
        y(:, k) = col; %#ok<AGROW>
    end
end

function y = prefix_cols(T, prefix, exclude_prefixes)
    if nargin < 3
        exclude_prefixes = {};
    end
    vars = T.Properties.VariableNames;
    mask = startsWith(vars, prefix);
    for k = 1:numel(exclude_prefixes)
        mask = mask & ~startsWith(vars, exclude_prefixes{k});
    end
    selected = vars(mask);
    if isempty(selected)
        y = [];
    else
        y = double(T{:, selected});
    end
end

function rpy = quat_wxyz_to_rpy(q)
    q = double(q);
    w = q(:,1); x = q(:,2); y = q(:,3); z = q(:,4);
    roll = atan2(2 .* (w .* x + y .* z), 1 - 2 .* (x.^2 + y.^2));
    pitch_arg = 2 .* (w .* y - z .* x);
    pitch_arg = max(min(pitch_arg, 1), -1);
    pitch = asin(pitch_arg);
    yaw = atan2(2 .* (w .* z + x .* y), 1 - 2 .* (y.^2 + z.^2));
    rpy = [roll, pitch, yaw];
end

function mark_done_lines(t, done)
    if isempty(t) || isempty(done)
        return;
    end
    idx = find(done(:) > 0.5);
    for k = 1:numel(idx)
        if idx(k) <= numel(t)
            xline(t(idx(k)), 'r:', 'HandleVisibility', 'off');
        end
    end
end

function plot_joint_grid(t, y1, y2, fig_name, y_label, name1, name2)
    if isempty(y1)
        return;
    end
    n = size(y1, 2);
    cols_n = 3;
    rows = ceil(n / cols_n);
    figure('Name', fig_name, 'NumberTitle', 'off', 'Position', [100 100 1550 900]);
    for j = 1:n
        subplot(rows, cols_n, j); hold on;
        plot(t, y1(:,j), 'LineWidth', 1.0, 'DisplayName', name1);
        if ~isempty(y2) && size(y2, 2) >= j
            plot(t, y2(:,j), '--', 'LineWidth', 1.0, 'DisplayName', name2);
            legend('Location', 'best');
        end
        grid on; xlabel('time [s]'); ylabel(y_label);
        title(sprintf('joint %d', j));
    end
    sgtitle(fig_name, 'Interpreter', 'none');
end
