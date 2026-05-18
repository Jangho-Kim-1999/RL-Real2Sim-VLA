% plot_play_gs_mat.m
% Plot MATLAB diagnostics saved by scripts/rsl_rl/play_gs.py --log_mat.
%
% Usage from MATLAB:
%   cd('.../VR-Robo-main/vrrobo_isaaclab')
%   plot_play_gs_mat
%
% Or choose a file explicitly before running:
%   MAT_PATH = 'logs/rsl_rl/mclquad_flat_cone_camera/2026-05-12_22-40-59/matlab_logs/play_model_1600.mat';
%   run('scripts/plot_play_gs_mat.m')

if ~exist('MAT_PATH', 'var') || isempty(MAT_PATH)
    MAT_PATH = find_latest_play_mat();
end

S = load(MAT_PATH);
fprintf('[INFO] Loaded: %s\n', MAT_PATH);

t = vec(get_or(S, 't_policy_s', []));
if isempty(t)
    n = size(get_or(S, 'velocity_command_b', []), 1);
    policy_dt = scalar_or(S, 'policy_dt', 1.0);
    t = (0:n-1)' * policy_dt;
end

sim_dt = scalar_or(S, 'sim_dt', NaN);
policy_dt = scalar_or(S, 'policy_dt', NaN);
low_level_dt = scalar_or(S, 'low_level_dt', NaN);
fprintf('[INFO] sim_dt=%.6g, policy_dt=%.6g, low_level_dt=%.6g\n', sim_dt, policy_dt, low_level_dt);

cmd = mat_or(S, 'velocity_command_b');
raw_action = mat_or(S, 'policy_action_raw');
root_v_b = mat_or(S, 'root_lin_vel_b');
root_w_b = mat_or(S, 'root_ang_vel_b');
root_pos_w = mat_or(S, 'root_pos_w');
root_quat_w = mat_or(S, 'root_quat_w');
reward = vec(get_or(S, 'reward', []));
done = vec(get_or(S, 'done', []));
cone_env = mat_or(S, 'cone_red_pos_env');
q = mat_or(S, 'joint_pos');
dq = mat_or(S, 'joint_vel');
q_des = mat_or(S, 'joint_pos_target');

joint_names = default_joint_names(size(q, 2));
if isfield(S, 'joint_names')
    joint_names = cellstr(string(S.joint_names(:)));
end

% -------------------------------------------------------------------------
% Figure 1: High-level command tracking
% -------------------------------------------------------------------------
if has_cols(cmd, 3) && has_cols(root_v_b, 2) && has_cols(root_w_b, 3)
    figure('Name', 'High-level Command Tracking', 'NumberTitle', 'off', 'Position', [60 60 1250 520]);
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
% Figure 2: Policy raw output and scaled command
% -------------------------------------------------------------------------
if has_cols(raw_action, 3) || has_cols(cmd, 3)
    figure('Name', 'Policy Output and Velocity Command', 'NumberTitle', 'off', 'Position', [80 80 1250 620]);
    names = {'x', 'y', 'yaw'};
    for k = 1:3
        subplot(2, 3, k); hold on;
        if has_cols(raw_action, k)
            plot(t, raw_action(:,k), 'LineWidth', 1.1);
        end
        yline(0, 'k:'); yline(1, 'k:'); yline(-1, 'k:');
        xlabel('time [s]'); ylabel('raw action'); title(['raw action ', names{k}]); grid on;

        subplot(2, 3, 3+k); hold on;
        if has_cols(cmd, k)
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
if has_cols(root_pos_w, 2)
    figure('Name', 'Trajectory', 'NumberTitle', 'off', 'Position', [120 120 780 700]);
    plot(root_pos_w(:,1), root_pos_w(:,2), 'LineWidth', 1.4, 'DisplayName', 'robot root'); hold on;
    if has_cols(cone_env, 2)
        plot(cone_env(:,1), cone_env(:,2), 'r--', 'LineWidth', 1.0, 'DisplayName', 'cone relative to env');
        scatter(cone_env(1,1), cone_env(1,2), 80, 'r', 'filled', 'DisplayName', 'cone start');
    end
    scatter(root_pos_w(1,1), root_pos_w(1,2), 60, 'g', 'filled', 'DisplayName', 'start');
    scatter(root_pos_w(end,1), root_pos_w(end,2), 60, 'k', 'filled', 'DisplayName', 'end');
    axis equal; grid on; xlabel('x [m]'); ylabel('y [m]');
    title('Robot trajectory');
    legend('Location', 'best');
end

% -------------------------------------------------------------------------
% Figure 4: Distance to cone, reward, done
% -------------------------------------------------------------------------
if has_cols(root_pos_w, 3) || ~isempty(reward)
    figure('Name', 'Task Signals', 'NumberTitle', 'off', 'Position', [140 140 1050 720]);
    subplot(3, 1, 1); hold on;
    if has_cols(root_pos_w, 3) && has_cols(cone_env, 3)
        dist_xy = vecnorm(root_pos_w(:,1:2) - cone_env(:,1:2), 2, 2);
        dist_3d = vecnorm(root_pos_w(:,1:3) - cone_env(:,1:3), 2, 2);
        plot(t, dist_xy, 'LineWidth', 1.2, 'DisplayName', 'xy distance');
        plot(t, dist_3d, 'LineWidth', 1.2, 'DisplayName', '3d distance');
        yline(0.35, 'k--', 'reach threshold');
        legend('Location', 'best');
    end
    mark_done_lines(t, done); grid on; xlabel('time [s]'); ylabel('distance [m]'); title('Distance to cone');

    subplot(3, 1, 2); hold on;
    if ~isempty(reward)
        plot(t, reward, 'LineWidth', 1.1);
    end
    mark_done_lines(t, done); grid on; xlabel('time [s]'); ylabel('reward'); title('Reward');

    subplot(3, 1, 3); hold on;
    if ~isempty(done)
        stairs(t, double(done), 'LineWidth', 1.1);
    end
    grid on; xlabel('time [s]'); ylabel('done'); title('Done flags');
end

% -------------------------------------------------------------------------
% Figure 5: Base position, orientation, velocity
% -------------------------------------------------------------------------
if has_cols(root_pos_w, 3) || has_cols(root_quat_w, 4)
    rpy = [];
    if has_cols(root_quat_w, 4)
        rpy = quat_wxyz_to_rpy(root_quat_w);
    end
    xyz_names = {'x', 'y', 'z'};
    rpy_names = {'roll', 'pitch', 'yaw'};
    figure('Name', 'Base State', 'NumberTitle', 'off', 'Position', [160 160 1250 800]);
    for k = 1:3
        subplot(3, 3, k); hold on;
        if has_cols(root_pos_w, k), plot(t, root_pos_w(:,k), 'LineWidth', 1.0); end
        grid on; xlabel('time [s]'); ylabel('m'); title(sprintf('root pos %s', xyz_names{k}));

        subplot(3, 3, 3+k); hold on;
        if has_cols(rpy, k), plot(t, rpy(:,k), 'LineWidth', 1.0); end
        grid on; xlabel('time [s]'); ylabel('rad'); title(sprintf('base %s', rpy_names{k}));

        subplot(3, 3, 6+k); hold on;
        if has_cols(root_v_b, k), plot(t, root_v_b(:,k), 'LineWidth', 1.0); end
        grid on; xlabel('time [s]'); ylabel('m/s'); title(sprintf('body linear vel %s', xyz_names{k}));
    end
    sgtitle('Base pose and velocity');
end

% -------------------------------------------------------------------------
% Figure 6: Joint position actual vs target at policy step
% -------------------------------------------------------------------------
if has_cols(q, 1)
    plot_joint_grid(t, q, q_des, joint_names, 'Joint Positions at Policy Step', 'rad', 'actual', 'target');
end

% -------------------------------------------------------------------------
% Figure 7: Joint velocity at policy step
% -------------------------------------------------------------------------
if has_cols(dq, 1)
    plot_joint_grid(t, dq, [], joint_names, 'Joint Velocities at Policy Step', 'rad/s', 'actual', '');
end

% -------------------------------------------------------------------------
% Figure 8: Physics-step low-level diagnostics
% -------------------------------------------------------------------------
phys_step = vec(get_or(S, 'phys_step', []));
if ~isempty(phys_step) && isfinite(sim_dt)
    t_phys = double(phys_step) * sim_dt;
    q_phys = mat_or(S, 'phys_joint_pos');
    q_des_phys = mat_or(S, 'phys_joint_pos_target');
    dq_phys = mat_or(S, 'phys_joint_vel');
    tau_applied = mat_or(S, 'phys_tau_applied');
    tau_computed = mat_or(S, 'phys_tau_computed');

    if has_cols(q_phys, 1)
        plot_joint_grid(t_phys, q_phys, q_des_phys, joint_names, 'Physics-step Joint Positions', 'rad', 'actual', 'target');
    end
    if has_cols(dq_phys, 1)
        plot_joint_grid(t_phys, dq_phys, [], joint_names, 'Physics-step Joint Velocities', 'rad/s', 'actual', '');
    end
    if has_cols(tau_applied, 1)
        plot_joint_grid(t_phys, tau_applied, tau_computed, joint_names, 'Physics-step Torques', 'Nm', 'applied', 'computed');
    end
end

% -------------------------------------------------------------------------
% Figure 9: Low-level policy output and observation heatmap
% -------------------------------------------------------------------------
ll_step = vec(get_or(S, 'll_update_phys_step', []));
if ~isempty(ll_step) && isfinite(sim_dt)
    t_ll = double(ll_step) * sim_dt;
    ll_output = mat_or(S, 'll_output');
    ll_obs = mat_or(S, 'll_obs');
    if has_cols(ll_output, 1)
        plot_joint_grid(t_ll, ll_output, [], joint_names, 'Low-level Policy Output', 'raw', 'output', '');
    end
    if has_cols(ll_obs, 1)
        figure('Name', 'Low-level Observation Heatmap', 'NumberTitle', 'off', 'Position', [180 180 1250 700]);
        imagesc(t_ll, 1:size(ll_obs,2), ll_obs');
        axis tight; colorbar; colormap(jet);
        xlabel('time [s]'); ylabel('LL obs index'); title('Low-level observation heatmap');
    end
end

fprintf('[DONE] Generated plots from: %s\n', MAT_PATH);

% =========================================================================
% Local helper functions
% =========================================================================
function path = find_latest_play_mat()
    files = dir(fullfile('logs', 'rsl_rl', '**', 'matlab_logs', '*.mat'));
    if isempty(files)
        error('No play .mat files found under logs/rsl_rl/**/matlab_logs. Set MAT_PATH manually.');
    end
    [~, idx] = max([files.datenum]);
    path = fullfile(files(idx).folder, files(idx).name);
end

function y = get_or(S, name, default_value)
    if isfield(S, name)
        y = S.(name);
    else
        y = default_value;
    end
end

function y = scalar_or(S, name, default_value)
    y = get_or(S, name, default_value);
    if isempty(y)
        y = default_value;
    else
        y = double(y(1));
    end
end

function y = vec(x)
    if isempty(x)
        y = [];
    else
        y = double(x(:));
    end
end

function y = mat_or(S, name)
    if isfield(S, name) && ~isempty(S.(name))
        y = double(squeeze(S.(name)));
        if isvector(y)
            y = y(:);
        end
    else
        y = [];
    end
end

function tf = has_cols(x, n)
    tf = ~isempty(x) && ismatrix(x) && size(x, 2) >= n;
end

function names = default_joint_names(n)
    base = {'FLHAA','FLHIP','FLKNEE','FRHAA','FRHIP','FRKNEE', ...
            'RLHAA','RLHIP','RLKNEE','RRHAA','RRHIP','RRKNEE'};
    if n <= numel(base)
        names = base(1:n);
    else
        names = cell(1, n);
        for k = 1:n
            names{k} = sprintf('joint_%02d', k);
        end
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

function plot_joint_grid(t, y1, y2, joint_names, fig_name, y_label, name1, name2)
    if isempty(y1)
        return;
    end
    n = size(y1, 2);
    cols = 3;
    rows = ceil(n / cols);
    figure('Name', fig_name, 'NumberTitle', 'off', 'Position', [100 100 1550 900]);
    for j = 1:n
        subplot(rows, cols, j); hold on;
        plot(t, y1(:,j), 'LineWidth', 1.0, 'DisplayName', name1);
        if ~isempty(y2) && size(y2, 2) >= j
            plot(t, y2(:,j), '--', 'LineWidth', 1.0, 'DisplayName', name2);
            legend('Location', 'best');
        end
        grid on; xlabel('time [s]'); ylabel(y_label);
        if j <= numel(joint_names)
            title(joint_names{j}, 'Interpreter', 'none');
        else
            title(sprintf('joint %d', j));
        end
    end
    sgtitle(fig_name, 'Interpreter', 'none');
end
