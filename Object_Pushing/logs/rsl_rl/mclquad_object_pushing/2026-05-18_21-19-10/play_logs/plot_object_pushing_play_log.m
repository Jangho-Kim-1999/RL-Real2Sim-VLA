clear; clc; close all;
log_file = fullfile(fileparts(mfilename('fullpath')), 'object_pushing_play_log.csv');
T = readtable(log_file);

figure('Name', 'Object Pushing Play Log', 'Color', 'w');
tiledlayout(2, 2);

nexttile;
plot(T.time_s, T.object_target_dist_xy, 'LineWidth', 1.5); hold on;
yline(0.05, '--r', 'success threshold');
grid on;
xlabel('time [s]');
ylabel('object-target XY dist [m]');
title('Position Error');

nexttile;
plot(T.object_x, T.object_y, 'b', 'LineWidth', 1.5); hold on;
plot(T.target_x, T.target_y, 'g--', 'LineWidth', 1.2);
plot(T.robot_x, T.robot_y, 'k:', 'LineWidth', 1.0);
axis equal; grid on;
xlabel('x [m]');
ylabel('y [m]');
legend('object', 'target', 'robot', 'Location', 'best');
title('XY Trajectory');

nexttile;
plot(T.time_s, T.reward, 'LineWidth', 1.2); grid on;
xlabel('time [s]');
ylabel('reward');
title('Reward');

nexttile;
plot(T.time_s, [T.velocity_cmd_vx, T.velocity_cmd_vy, T.velocity_cmd_wz], 'LineWidth', 1.1);
grid on;
xlabel('time [s]');
ylabel('command');
legend('v_x', 'v_y', 'w_z', 'Location', 'best');
title('High-Level Velocity Command');
