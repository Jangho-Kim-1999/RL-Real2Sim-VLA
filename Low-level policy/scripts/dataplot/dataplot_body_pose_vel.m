close all; clear; clc;

% Body-pose velocity command tracking plot for logs/data.csv.
% Current command semantics:
%   cmd[0] = vx_ref          -> track with base_lin_vel_x
%   cmd[1] = vz_ref          -> track with base_lin_vel_z
%   cmd[2] = pitch_rate_ref  -> track with base_ang_vel_y
%
% NOTE:
% play.py header can be either:
%   - new body_pose:    cmd_lin_vel_x, cmd_lin_vel_z, cmd_ang_vel_pitch
%   - legacy/generic:   cmd_lin_vel_x, cmd_lin_vel_y, cmd_ang_vel_z
%   - generic indexed:  cmd_0, cmd_1, cmd_2
% This script accepts all variants and maps to current semantics.

%% User settings
csv_path = "logs/data.csv";
default_policy_hz = 55.56;
plot_time_start_s = 0.0;
plot_time_end_s = inf;
z_ref_initial = 0.3536;
pitch_ref_initial = 0.0;
x_ref_initial = 0.0;

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

%% Resolve columns (robust to legacy/new header names)
cmd_vx_col = pick_first_existing(vars, ["cmd_lin_vel_x", "cmd_0"]);
cmd_vz_col = pick_first_existing(vars, ["cmd_lin_vel_z", "cmd_lin_vel_y", "cmd_1"]);
cmd_pitch_rate_col = pick_first_existing(vars, ["cmd_ang_vel_pitch", "cmd_ang_vel_z", "cmd_2"]);

act_vx_col = pick_first_existing(vars, ["base_lin_vel_x"]);
act_vz_col = pick_first_existing(vars, ["base_lin_vel_z", "base_lin_vel_y"]);
act_pitch_rate_col = pick_first_existing(vars, ["base_ang_vel_y", "base_ang_vel_z"]);

required_cols = [cmd_vx_col, cmd_vz_col, cmd_pitch_rate_col, act_vx_col, act_vz_col, act_pitch_rate_col];
if any(strlength(required_cols) == 0)
    fprintf("[ERROR] Missing required columns for body-pose velocity tracking.\n");
    fprintf("[INFO] Required command candidates:\n");
    fprintf("  vx_ref: cmd_lin_vel_x or cmd_0\n");
    fprintf("  vz_ref: cmd_lin_vel_z or cmd_lin_vel_y or cmd_1\n");
    fprintf("  pitch_rate_ref: cmd_ang_vel_pitch or cmd_ang_vel_z or cmd_2\n");
    fprintf("[INFO] Required state candidates:\n");
    fprintf("  vx: base_lin_vel_x\n");
    fprintf("  vz: base_lin_vel_z (fallback: base_lin_vel_y)\n");
    fprintf("  pitch_rate: base_ang_vel_y (fallback: base_ang_vel_z)\n");
    return;
end

cmd_vx = T{:, cmd_vx_col};
cmd_vz = T{:, cmd_vz_col};
cmd_pitch_rate = T{:, cmd_pitch_rate_col};

act_vx = T{:, act_vx_col};
act_vz = T{:, act_vz_col};
act_pitch_rate = T{:, act_pitch_rate_col};

rmse_vx = sqrt(mean((act_vx - cmd_vx).^2, "omitnan"));
rmse_vz = sqrt(mean((act_vz - cmd_vz).^2, "omitnan"));
rmse_pitch_rate = sqrt(mean((act_pitch_rate - cmd_pitch_rate).^2, "omitnan"));

% Position-like trajectories from integrating velocity commands/states.
cmd_z_ref_traj = z_ref_initial + cumtrapz(t, cmd_vz);
cmd_pitch_ref_traj = pitch_ref_initial + cumtrapz(t, cmd_pitch_rate);
cmd_x_ref_traj = x_ref_initial + cumtrapz(t, cmd_vx);

base_height_col = pick_first_existing(vars, ["base_height"]);
base_pitch_col = pick_first_existing(vars, ["base_pitch"]);
base_roll_col = pick_first_existing(vars, ["base_roll"]);
base_yaw_col = pick_first_existing(vars, ["base_yaw"]);
base_pos_x_col = pick_first_existing(vars, ["base_pos_x"]);
base_pos_y_col = pick_first_existing(vars, ["base_pos_y"]);
base_pos_z_col = pick_first_existing(vars, ["base_pos_z", "base_height"]);
base_lin_vel_x_col = pick_first_existing(vars, ["base_lin_vel_x"]);
base_lin_vel_y_col = pick_first_existing(vars, ["base_lin_vel_y"]);
base_lin_vel_z_col = pick_first_existing(vars, ["base_lin_vel_z"]);
base_ang_vel_x_col = pick_first_existing(vars, ["base_ang_vel_x"]);
base_ang_vel_y_col = pick_first_existing(vars, ["base_ang_vel_y"]);
base_ang_vel_z_col = pick_first_existing(vars, ["base_ang_vel_z"]);
has_base_height = strlength(base_height_col) > 0;
has_base_pitch = strlength(base_pitch_col) > 0;
has_base_roll = strlength(base_roll_col) > 0;
has_base_yaw = strlength(base_yaw_col) > 0;
has_base_pos_x = strlength(base_pos_x_col) > 0;
has_base_pos_y = strlength(base_pos_y_col) > 0;
has_base_pos_z = strlength(base_pos_z_col) > 0;
if has_base_height
    base_height = T{:, base_height_col};
else
    base_height = [];
end
if has_base_pitch
    base_pitch = T{:, base_pitch_col};
else
    base_pitch = [];
end
if has_base_roll
    base_roll = T{:, base_roll_col};
else
    base_roll = [];
end
if has_base_yaw
    base_yaw = T{:, base_yaw_col};
else
    base_yaw = [];
end
if has_base_pos_x
    base_pos_x = T{:, base_pos_x_col};
else
    base_pos_x = [];
end
if has_base_pos_y
    base_pos_y = T{:, base_pos_y_col};
else
    base_pos_y = [];
end
if has_base_pos_z
    base_pos_z = T{:, base_pos_z_col};
else
    base_pos_z = [];
end

%% Plot
f = figure("Name", "Body Pose Velocity Tracking", "Color", "w");
tiledlayout(3, 1, "TileSpacing", "compact", "Padding", "compact");

ax1 = nexttile;
h11 = plot(ax1, t, cmd_vx, "LineWidth", 1.1, "Color", [0.85, 0.33, 0.10]); hold(ax1, "on");
h12 = plot(ax1, t, act_vx, "LineWidth", 1.1, "Color", [0.00, 0.45, 0.74]); hold(ax1, "off");
grid(ax1, "on");
ylabel(ax1, "v_x [m/s]");
title(ax1, sprintf("x-velocity tracking (RMSE = %.4f)", rmse_vx));
lg = legend(ax1, [h11, h12], {"command", "actual"}, "Location", "best");
lg.AutoUpdate = "off";

ax2 = nexttile;
plot(ax2, t, cmd_vz, "LineWidth", 1.1, "Color", [0.85, 0.33, 0.10]); hold(ax2, "on");
plot(ax2, t, act_vz, "LineWidth", 1.1, "Color", [0.00, 0.45, 0.74]); hold(ax2, "off");
grid(ax2, "on");
ylabel(ax2, "v_z [m/s]");
title(ax2, sprintf("z-velocity tracking (RMSE = %.4f)", rmse_vz));

ax3 = nexttile;
plot(ax3, t, cmd_pitch_rate, "LineWidth", 1.1, "Color", [0.85, 0.33, 0.10]); hold(ax3, "on");
plot(ax3, t, act_pitch_rate, "LineWidth", 1.1, "Color", [0.00, 0.45, 0.74]); hold(ax3, "off");
grid(ax3, "on");
ylabel(ax3, "\omega_{pitch} [rad/s]");
xlabel(ax3, "t [s]");
title(ax3, sprintf("pitch-rate tracking (RMSE = %.4f)", rmse_pitch_rate));

linkaxes([ax1, ax2, ax3], "x");

% Integrated trajectories (position-domain view)
f2 = figure("Name", "Body Pose Position Trajectory (Integrated)", "Color", "w");
tiledlayout(3, 1, "TileSpacing", "compact", "Padding", "compact");

ax0 = nexttile;
h01 = plot(ax0, t, cmd_x_ref_traj, "LineWidth", 1.1, "Color", [0.85, 0.33, 0.10]); hold(ax0, "on");
legend_items0 = h01;
legend_names0 = {"cmd-integrated x_ref"};
if has_base_pos_x
    h02 = plot(ax0, t, base_pos_x, "--", "LineWidth", 1.0, "Color", [0.20, 0.20, 0.20]);
    legend_items0(end+1) = h02; %#ok<AGROW>
    legend_names0{end+1} = "base_pos_x (raw)"; %#ok<AGROW>
end
hold(ax0, "off");
grid(ax0, "on");
ylabel(ax0, "x [m]");
title(ax0, "x trajectory from command velocity integration");
lg1 = legend(ax0, legend_items0, legend_names0, "Location", "best");
lg1.AutoUpdate = "off";

ax4 = nexttile;
h41 = plot(ax4, t, cmd_z_ref_traj, "LineWidth", 1.1, "Color", [0.85, 0.33, 0.10]); hold(ax4, "on");
legend_items = h41;
legend_names = {"cmd-integrated z_ref"};
if has_base_height
    h43 = plot(ax4, t, base_height, "--", "LineWidth", 1.0, "Color", [0.20, 0.20, 0.20]);
    legend_items(end+1) = h43; %#ok<AGROW>
    legend_names{end+1} = "base_height (raw)"; %#ok<AGROW>
end
hold(ax4, "off");
grid(ax4, "on");
ylabel(ax4, "z [m]");
title(ax4, "z trajectory from command velocity integration");
lg2 = legend(ax4, legend_items, legend_names, "Location", "best");
lg2.AutoUpdate = "off";

ax5 = nexttile;
h51 = plot(ax5, t, cmd_pitch_ref_traj, "LineWidth", 1.1, "Color", [0.85, 0.33, 0.10]); hold(ax5, "on");
legend_items2 = h51;
legend_names2 = {"cmd-integrated pitch_ref"};
if has_base_pitch
    h53 = plot(ax5, t, base_pitch, "--", "LineWidth", 1.0, "Color", [0.20, 0.20, 0.20]);
    legend_items2(end+1) = h53; %#ok<AGROW>
    legend_names2{end+1} = "base_pitch (raw)"; %#ok<AGROW>
end
hold(ax5, "off");
grid(ax5, "on");
ylabel(ax5, "pitch [rad]");
xlabel(ax5, "t [s]");
title(ax5, "pitch trajectory from command angular-velocity integration");
lg3 = legend(ax5, legend_items2, legend_names2, "Location", "best");
lg3.AutoUpdate = "off";

linkaxes([ax0, ax4, ax5], "x");

% Raw base state plot (position/velocity/RPY/angular velocity)
if strlength(base_lin_vel_x_col) > 0 && strlength(base_lin_vel_y_col) > 0 && strlength(base_lin_vel_z_col) > 0 && ...
   strlength(base_ang_vel_x_col) > 0 && strlength(base_ang_vel_y_col) > 0 && strlength(base_ang_vel_z_col) > 0 && ...
   has_base_roll && has_base_pitch && has_base_yaw && has_base_pos_x && has_base_pos_y && has_base_pos_z
    base_lin_vel_x = T{:, base_lin_vel_x_col};
    base_lin_vel_y = T{:, base_lin_vel_y_col};
    base_lin_vel_z = T{:, base_lin_vel_z_col};
    base_ang_vel_x = T{:, base_ang_vel_x_col};
    base_ang_vel_y = T{:, base_ang_vel_y_col};
    base_ang_vel_z = T{:, base_ang_vel_z_col};

    f3 = figure("Name", "Base State (Raw)", "Color", "w");
    tiledlayout(4, 3, "TileSpacing", "compact", "Padding", "compact");

    nexttile; plot(t, base_pos_x, "LineWidth", 1.0); grid on; ylabel("x [m]"); title("base pos x");
    nexttile; plot(t, base_pos_y, "LineWidth", 1.0); grid on; ylabel("y [m]"); title("base pos y");
    nexttile; plot(t, base_pos_z, "LineWidth", 1.0); grid on; ylabel("z [m]"); title("base pos z");

    nexttile; plot(t, base_lin_vel_x, "LineWidth", 1.0); grid on; ylabel("v_x [m/s]"); title("base lin vel x");
    nexttile; plot(t, base_lin_vel_y, "LineWidth", 1.0); grid on; ylabel("v_y [m/s]"); title("base lin vel y");
    nexttile; plot(t, base_lin_vel_z, "LineWidth", 1.0); grid on; ylabel("v_z [m/s]"); title("base lin vel z");

    nexttile; plot(t, base_roll, "LineWidth", 1.0); grid on; ylabel("roll [rad]"); title("base roll");
    nexttile; plot(t, base_pitch, "LineWidth", 1.0); grid on; ylabel("pitch [rad]"); title("base pitch");
    nexttile; plot(t, base_yaw, "LineWidth", 1.0); grid on; ylabel("yaw [rad]"); title("base yaw");

    nexttile; plot(t, base_ang_vel_x, "LineWidth", 1.0); grid on; ylabel("\omega_x [rad/s]"); title("base ang vel x"); xlabel("t [s]");
    nexttile; plot(t, base_ang_vel_y, "LineWidth", 1.0); grid on; ylabel("\omega_y [rad/s]"); title("base ang vel y"); xlabel("t [s]");
    nexttile; plot(t, base_ang_vel_z, "LineWidth", 1.0); grid on; ylabel("\omega_z [rad/s]"); title("base ang vel z"); xlabel("t [s]");
else
    fprintf("[WARN] Some raw base-state columns are missing. Skip raw base-state figure.\n");
end

fprintf("[INFO] Command column mapping:\n");
fprintf("  vx_ref         <- %s\n", cmd_vx_col);
fprintf("  vz_ref         <- %s\n", cmd_vz_col);
fprintf("  pitch_rate_ref <- %s\n", cmd_pitch_rate_col);
fprintf("[INFO] State column mapping:\n");
fprintf("  vx         <- %s\n", act_vx_col);
fprintf("  vz         <- %s\n", act_vz_col);
fprintf("  pitch_rate <- %s\n", act_pitch_rate_col);
if has_base_height
    fprintf("  base_height <- %s\n", base_height_col);
end
if has_base_pitch
    fprintf("  base_pitch  <- %s\n", base_pitch_col);
end
if has_base_pos_x
    fprintf("  base_pos_x  <- %s\n", base_pos_x_col);
end
if has_base_pos_y
    fprintf("  base_pos_y  <- %s\n", base_pos_y_col);
end
if has_base_pos_z
    fprintf("  base_pos_z  <- %s\n", base_pos_z_col);
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

function col = pick_first_existing(vars, candidates)
    col = "";
    candidates = string(candidates);
    for i = 1:numel(candidates)
        if any(vars == candidates(i))
            col = candidates(i);
            return;
        end
    end
end
