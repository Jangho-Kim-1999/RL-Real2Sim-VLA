% plot_ll_diagnostics.m
% Load ll_diagnostics.mat and generate diagnostic plots.
% Usage:
%   cd('.../vrrobo_isaaclab/scripts')
%   plot_ll_diagnostics

if ~exist('MAT_PATH', 'var')
    MAT_PATH = 'logs/ll_diag/ll_diagnostics.mat';
end
if ~exist('ENV_IDX', 'var')
    ENV_IDX  = 1;   % MATLAB 1-based index (env 0 in Python = 1 here)
end

d = load(MAT_PATH);

sim_dt = double(d.sim_dt);
rl_dt  = double(d.rl_dt);

N_rl  = size(d.vel_cmd, 1);
t_rl  = (0:N_rl-1)' * rl_dt;

if ~isempty(d.phys_step)
    t_phys = double(d.phys_step(:)) * sim_dt;
else
    t_phys = [];
end

if ~isempty(d.ll_phys_step)
    t_ll = double(d.ll_phys_step(:)) * sim_dt;
else
    t_ll = [];
end

reset_steps = find(d.reset_mask);  % 1-based MATLAB indices

if isfield(d, 'joint_names')
    joint_names = cellstr(string(d.joint_names));
else
    joint_names = {'FLHAA','FLHIP','FLKNEE','FRHAA','FRHIP','FRKNEE', ...
                   'RLHAA','RLHIP','RLKNEE','RRHAA','RRHIP','RRKNEE'};
end

% ---------------------------------------------------------
% Figure 1: Velocity tracking (env 0)
% ---------------------------------------------------------
vel_cmd_e0  = squeeze(d.vel_cmd(:, ENV_IDX, :));
vel_meas_e0 = squeeze(d.vel_measured(:, ENV_IDX, :));
labels_v    = {'vx (m/s)', 'vy (m/s)', 'wz (rad/s)'};

figure('Name', 'Velocity Tracking', 'NumberTitle', 'off', 'Position', [50 50 1200 500]);
for i = 1:3
    subplot(1, 3, i); hold on;
    plot(t_rl, vel_cmd_e0(:,i),  'b--', 'LineWidth', 1.5, 'DisplayName', 'Command');
    plot(t_rl, vel_meas_e0(:,i), 'r',   'LineWidth', 1.2, 'DisplayName', 'Measured');
    mark_resets(t_rl, reset_steps);
    xlabel('Time (s)'); ylabel(labels_v{i});
    title(labels_v{i}); legend; grid on;
end
sgtitle('Velocity Tracking - env 0');

% ---------------------------------------------------------
% Figure 2: Base height and foot clearance (env 0)
% ---------------------------------------------------------
base_z_e0   = d.base_z(:, ENV_IDX);
min_foot_e0 = d.min_foot_z(:, ENV_IDX);

figure('Name', 'Base Height', 'NumberTitle', 'off', 'Position', [50 600 800 400]);
subplot(2, 1, 1); hold on;
plot(t_rl, base_z_e0, 'b', 'LineWidth', 1.2);
yline(0.35, 'k--', 'target 0.35m');
mark_resets(t_rl, reset_steps);
xlabel('Time (s)'); ylabel('z (m)'); title('Base Height'); grid on;

subplot(2, 1, 2); hold on;
plot(t_rl, min_foot_e0, 'r', 'LineWidth', 1.2);
yline(0, 'k--', 'ground');
mark_resets(t_rl, reset_steps);
xlabel('Time (s)'); ylabel('z (m)'); title('Min Foot Clearance'); grid on;
sgtitle('Base Height and Min Foot Clearance - env 0');

% ---------------------------------------------------------
% Figure 3: Joint positions actual vs desired (phys step, env 0)
% ---------------------------------------------------------
if ~isempty(t_phys) && ~isempty(d.q_phys)
    figure('Name', 'Joint Positions', 'NumberTitle', 'off', 'Position', [100 50 1600 900]);
    for j = 1:12
        subplot(4, 3, j); hold on;
        plot(t_phys, d.q_phys(:,j),    'b',   'LineWidth', 1.0, 'DisplayName', 'Actual');
        plot(t_phys, d.q_des_phys(:,j), 'r--', 'LineWidth', 1.0, 'DisplayName', 'Desired');
        xlabel('Time (s)'); ylabel('rad');
        title(joint_names{j}); legend('Location', 'best'); grid on;
    end
    sgtitle('Joint Positions: Actual vs Desired (env 0)');
end

% ---------------------------------------------------------
% Figure 4: Torques (phys step, env 0)
% ---------------------------------------------------------
if ~isempty(t_phys) && ~isempty(d.tau_applied)
    figure('Name', 'Torques', 'NumberTitle', 'off', 'Position', [100 50 1600 900]);
    for j = 1:12
        subplot(4, 3, j); hold on;
        plot(t_phys, d.tau_computed(:,j), 'b',   'LineWidth', 1.0, 'DisplayName', 'Computed');
        plot(t_phys, d.tau_applied(:,j),  'r--', 'LineWidth', 1.0, 'DisplayName', 'Applied');
        yline( 60, 'k:', '+60Nm');
        yline(-60, 'k:', '-60Nm');
        xlabel('Time (s)'); ylabel('Nm');
        title(joint_names{j}); legend('Location', 'best'); grid on;
    end
    sgtitle('Torques: Computed vs Applied after delay (env 0)');
end

% ---------------------------------------------------------
% Figure 5: Joint velocities (phys step, env 0)
% ---------------------------------------------------------
if ~isempty(t_phys) && ~isempty(d.dq_phys)
    figure('Name', 'Joint Velocities', 'NumberTitle', 'off', 'Position', [100 50 1600 900]);
    for j = 1:12
        subplot(4, 3, j); hold on;
        plot(t_phys, d.dq_phys(:,j), 'b', 'LineWidth', 1.0);
        xlabel('Time (s)'); ylabel('rad/s');
        title(joint_names{j}); grid on;
    end
    sgtitle('Joint Velocities actual (env 0)');
end

% ---------------------------------------------------------
% Figure 6: LL policy raw output (LL step, env 0)
% ---------------------------------------------------------
if ~isempty(t_ll) && ~isempty(d.ll_output)
    figure('Name', 'LL Policy Output', 'NumberTitle', 'off', 'Position', [100 50 1600 900]);
    for j = 1:12
        subplot(4, 3, j); hold on;
        plot(t_ll, d.ll_output(:,j), 'b', 'LineWidth', 1.0);
        yline( 1, 'k:');
        yline(-1, 'k:');
        xlabel('Time (s)'); ylabel('raw');
        title(joint_names{j}); grid on;
    end
    sgtitle('LL Policy Raw Output before scale/offset (env 0)');
end

% ---------------------------------------------------------
% Figure 7a: LL obs heatmap (45D, LL step, env 0)
% ---------------------------------------------------------
if ~isempty(t_ll) && ~isempty(d.ll_obs)
    obs_labels = cell(1, 45);
    for k = 1:3;  obs_labels{k}    = sprintf('angvel_%d', k);    end
    for k = 1:3;  obs_labels{3+k}  = sprintf('grav_%d', k);      end
    for k = 1:3;  obs_labels{6+k}  = sprintf('velcmd_%d', k);    end
    for k = 1:12; obs_labels{9+k}  = sprintf('jpos_%s', joint_names{k});  end
    for k = 1:12; obs_labels{21+k} = sprintf('jvel_%s', joint_names{k});  end
    for k = 1:12; obs_labels{33+k} = sprintf('llact_%s', joint_names{k}); end

    figure('Name', 'LL Obs Heatmap', 'NumberTitle', 'off', 'Position', [50 50 1400 700]);
    imagesc(t_ll, 1:45, d.ll_obs');
    colorbar; colormap(jet);
    set(gca, 'YTick', 1:45, 'YTickLabel', obs_labels, 'FontSize', 7);
    xlabel('Time (s)');
    title('LL Observation 45D over time - env 0');
end

% ---------------------------------------------------------
% Figure 7b: LL obs by block (LL step, env 0)
% ---------------------------------------------------------
if ~isempty(t_ll) && ~isempty(d.ll_obs)
    block_ranges = {1:3, 4:6, 7:9, 10:21, 22:33, 34:45};
    block_titles = {'ang_vel x0.25 (idx 1-3)', ...
                    'proj_grav (idx 4-6)', ...
                    'vel_cmd (idx 7-9)', ...
                    'joint_pos_bispace (idx 10-21)', ...
                    'joint_vel_bispace x0.05 (idx 22-33)', ...
                    'll_last_action (idx 34-45)'};

    figure('Name', 'LL Obs Blocks', 'NumberTitle', 'off', 'Position', [50 50 1400 900]);
    for b = 1:6
        subplot(6, 1, b); hold on;
        plot(t_ll, d.ll_obs(:, block_ranges{b}));
        xlabel('Time (s)');
        title(block_titles{b});
        grid on;
        xlim([t_ll(1), t_ll(end)]);
    end
    sgtitle('LL Observation Blocks - env 0');
end

% ---------------------------------------------------------
% Figure 8: LL last-action fed back (RL step, env 0)
% ---------------------------------------------------------
figure('Name', 'LL Last Action', 'NumberTitle', 'off', 'Position', [100 50 1200 600]);
for j = 1:12
    subplot(4, 3, j); hold on;
    plot(t_rl, d.raw_actions_env0(:,j), 'b', 'LineWidth', 1.0);
    yline( 1, 'k:');
    yline(-1, 'k:');
    mark_resets(t_rl, reset_steps);
    xlabel('Time (s)'); ylabel('raw');
    title(joint_names{j}); grid on;
end
sgtitle('LL raw output fed back as ll\_last\_action (env 0, RL step)');

fprintf('[DONE] All figures generated from: %s\n', MAT_PATH);
