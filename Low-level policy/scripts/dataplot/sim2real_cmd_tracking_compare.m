close all; clear; clc;

% Sim2Real command-speed tracking comparison.
% - Sim source:  logs/data.csv
% - Real source: scripts/dataplot/260429_exp/*.csv
%
% This script compares trunk/body speed command tracking only.
% It auto-detects columns and gracefully skips unavailable axes.

%% User settings
sim_csv_rel = "logs/data.csv";
real_csv_rel = "scripts/dataplot/260429_exp/260429_Proposed_1.5ms.csv";
sim_time_range_s = [0, inf];
real_time_range_s = [0, inf];
default_hz_if_no_time = 55.56;

%% Resolve paths and load tables
script_dir = fileparts(mfilename("fullpath"));
project_root = fileparts(fileparts(script_dir));
sim_csv = fullfile(project_root, sim_csv_rel);
real_csv = fullfile(project_root, real_csv_rel);

assert(isfile(sim_csv), "SIM csv not found: %s", sim_csv);
assert(isfile(real_csv), "REAL csv not found: %s", real_csv);

T_sim = readtable(sim_csv, "VariableNamingRule", "preserve");
T_real = readtable(real_csv, "VariableNamingRule", "preserve");

[t_sim, T_sim] = extract_time_and_crop(T_sim, sim_time_range_s, default_hz_if_no_time);
[t_real, T_real] = extract_time_and_crop(T_real, real_time_range_s, default_hz_if_no_time);

sim_vars = string(T_sim.Properties.VariableNames);
real_vars = string(T_real.Properties.VariableNames);
sim_vars_trim = strtrim(sim_vars);
real_vars_trim = strtrim(real_vars);

%% Column mapping
% Axis definitions (aligned to your current policy semantics):
%   axis 1: vx
%   axis 2: vz
%   axis 3: pitch rate
axis_names = ["v_x", "v_z", "\omega_{pitch}"];
axis_units = ["m/s", "m/s", "rad/s"];

sim_cmd_cols = [
    pick_first_existing(sim_vars, sim_vars_trim, ["cmd_lin_vel_x", "cmd_0"]), ...
    pick_first_existing(sim_vars, sim_vars_trim, ["cmd_lin_vel_z", "cmd_lin_vel_y", "cmd_1"]), ...
    pick_first_existing(sim_vars, sim_vars_trim, ["cmd_ang_vel_pitch", "cmd_ang_vel_z", "cmd_2"])
];
sim_act_cols = [
    pick_first_existing(sim_vars, sim_vars_trim, ["base_lin_vel_x"]), ...
    pick_first_existing(sim_vars, sim_vars_trim, ["base_lin_vel_z", "base_lin_vel_y"]), ...
    pick_first_existing(sim_vars, sim_vars_trim, ["base_ang_vel_y", "base_ang_vel_z"])
];

real_cmd_cols = [
    pick_first_existing(real_vars, real_vars_trim, ["policy_velocity_command_0", "cmd_lin_vel_x", "cmd_0"]), ...
    pick_first_existing(real_vars, real_vars_trim, ["policy_velocity_command_1", "cmd_lin_vel_z", "cmd_lin_vel_y", "cmd_1"]), ...
    pick_first_existing(real_vars, real_vars_trim, ["policy_velocity_command_2", "cmd_ang_vel_pitch", "cmd_ang_vel_z", "cmd_2"])
];
real_act_cols = [
    pick_first_existing(real_vars, real_vars_trim, ["body_lin_vel_0", "base_lin_vel_x", "body_vel_0"]), ...
    pick_first_existing(real_vars, real_vars_trim, ["body_lin_vel_2", "base_lin_vel_z", "body_vel_2"]), ...
    pick_first_existing(real_vars, real_vars_trim, ["body_rot_rate_1", "policy_base_ang_vel_1", "base_ang_vel_y", "base_ang_vel_z"])
];

%% Compute RMSE (when both command and actual are available)
rmse_sim = nan(1, 3);
rmse_real = nan(1, 3);

for k = 1:3
    if strlength(sim_cmd_cols(k)) > 0 && strlength(sim_act_cols(k)) > 0
        c = T_sim{:, sim_cmd_cols(k)};
        a = T_sim{:, sim_act_cols(k)};
        rmse_sim(k) = sqrt(mean((a - c).^2, "omitnan"));
    end

    if strlength(real_cmd_cols(k)) > 0 && strlength(real_act_cols(k)) > 0
        c = T_real{:, real_cmd_cols(k)};
        a = T_real{:, real_act_cols(k)};
        rmse_real(k) = sqrt(mean((a - c).^2, "omitnan"));
    end
end

%% Print summary
fprintf("\n=== Sim2Real Command Tracking Summary ===\n");
for k = 1:3
    fprintf("[%s] SIM  cmd=%s, act=%s, RMSE=%.5f\n", axis_names(k), ...
        str_or_na(sim_cmd_cols(k)), str_or_na(sim_act_cols(k)), rmse_sim(k));
    fprintf("[%s] REAL cmd=%s, act=%s, RMSE=%.5f\n", axis_names(k), ...
        str_or_na(real_cmd_cols(k)), str_or_na(real_act_cols(k)), rmse_real(k));
end
fprintf("=========================================\n\n");

%% Plot 1: Time-series (left SIM, right REAL)
figure("Name", "Sim2Real Command Tracking (Time Series)", "Color", "w");
tiledlayout(3, 2, "TileSpacing", "compact", "Padding", "compact");

for k = 1:3
    % SIM
    ax = nexttile((k-1)*2 + 1);
    hold(ax, "on"); grid(ax, "on");
    title(ax, sprintf("SIM %s", axis_names(k)));
    ylabel(ax, sprintf("%s [%s]", axis_names(k), axis_units(k)));
    if k == 3
        xlabel(ax, "t [s]");
    end
    if strlength(sim_cmd_cols(k)) > 0
        plot(ax, t_sim, T_sim{:, sim_cmd_cols(k)}, "LineWidth", 1.2, "DisplayName", "cmd");
    end
    if strlength(sim_act_cols(k)) > 0
        plot(ax, t_sim, T_sim{:, sim_act_cols(k)}, "LineWidth", 1.2, "DisplayName", "actual");
    else
        text(ax, 0.03, 0.85, "actual missing", "Units", "normalized", "Color", [0.8 0.2 0.2]);
    end
    legend(ax, "Location", "best");

    % REAL
    ax = nexttile((k-1)*2 + 2);
    hold(ax, "on"); grid(ax, "on");
    title(ax, sprintf("REAL %s", axis_names(k)));
    ylabel(ax, sprintf("%s [%s]", axis_names(k), axis_units(k)));
    if k == 3
        xlabel(ax, "t [s]");
    end
    if strlength(real_cmd_cols(k)) > 0
        plot(ax, t_real, T_real{:, real_cmd_cols(k)}, "LineWidth", 1.2, "DisplayName", "cmd");
    end
    if strlength(real_act_cols(k)) > 0
        plot(ax, t_real, T_real{:, real_act_cols(k)}, "LineWidth", 1.2, "DisplayName", "actual");
    else
        text(ax, 0.03, 0.85, "actual missing", "Units", "normalized", "Color", [0.8 0.2 0.2]);
    end
    legend(ax, "Location", "best");
end

%% Plot 2: RMSE bar comparison (only available axes)
figure("Name", "Sim2Real Command Tracking RMSE", "Color", "w");
rmse_mat = [rmse_sim(:), rmse_real(:)];
bar(rmse_mat);
grid on;
set(gca, "XTickLabel", cellstr(axis_names));
ylabel("RMSE");
legend({"SIM", "REAL"}, "Location", "best");
title("Command tracking RMSE (NaN = unavailable)");

%% Helpers
function [t, T] = extract_time_and_crop(T, range_s, default_hz)
vars = string(T.Properties.VariableNames);
vars_trim = strtrim(vars);
time_col = pick_first_existing(vars, vars_trim, ["time_s", "t"]);

if strlength(time_col) > 0
    t = T{:, time_col};
else
    step_col = pick_first_existing(vars, vars_trim, ["step"]);
    if strlength(step_col) > 0
        t = T{:, step_col} / default_hz;
    else
        t = (0:height(T)-1)' / default_hz;
    end
end

mask = (t >= range_s(1)) & (t <= range_s(2));
if ~any(mask)
    error("No samples in range [%.3f, %.3f].", range_s(1), range_s(2));
end
T = T(mask, :);
t = t(mask);
end

function col = pick_first_existing(orig_vars, trim_vars, candidates)
col = "";
for i = 1:numel(candidates)
    idx = find(trim_vars == candidates(i), 1, "first");
    if ~isempty(idx)
        col = orig_vars(idx);
        return;
    end
end
end

function out = str_or_na(s)
if strlength(s) == 0
    out = "N/A";
else
    out = s;
end
end

