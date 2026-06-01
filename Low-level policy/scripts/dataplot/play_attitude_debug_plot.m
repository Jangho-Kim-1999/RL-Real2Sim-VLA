% play_attitude_debug_plot.m
% Plot command vs simulated signals and action vs joint angles from play.py log CSV.
%
% Usage:
%   1) Run play with logging enabled (default writes logs/data.csv):
%      python3 scripts/play.py --task Attitude-MCLQuad-serial
%   2) In MATLAB:
%      play_attitude_debug_plot
%      % or
%      play_attitude_debug_plot('logs/data.csv')


function play_attitude_debug_plot(csv_path)
if nargin < 1 || strlength(string(csv_path)) == 0
    csv_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'logs', 'pitch_Dyn_NoRand.csv');
    csv_path = char(java.io.File(csv_path).getCanonicalPath());
end

if ~isfile(csv_path)
    error('CSV not found: %s', csv_path);
end

T = readtable(csv_path, 'VariableNamingRule', 'preserve');
vars = string(T.Properties.VariableNames);

if any(vars == "time_s")
    t = T.("time_s");
elseif any(vars == "step")
    t = T.("step");
else
    t = (0:height(T)-1).';
end
t = double(t);

cmd_cols = vars(startsWith(vars, "cmd_"));
action_cols = vars(startsWith(vars, "action_"));
joint_cols = vars(startsWith(vars, "joint_angle_"));
target_cols = vars(startsWith(vars, "target_angle_"));

if isempty(cmd_cols)
    warning('No command columns (cmd_*) found.');
end

% Figure 1: 12-DoF target angle vs joint angle
if isempty(joint_cols)
    warning('No joint_angle_* columns found. Skip joint-level figure.');
else
    joint_names = erase(joint_cols, "joint_angle_");
    n_joint = numel(joint_cols);
    n_col = 3;
    n_row = ceil(n_joint / n_col);

    f1 = figure('Name', '12-DoF Ref vs Joint Angle', 'Color', 'w');
    tiledlayout(f1, n_row, n_col, 'Padding', 'compact', 'TileSpacing', 'compact');

    for k = 1:n_joint
        nexttile;
        jn = joint_names(k);
        jcol = "joint_angle_" + jn;
        acol = "action_" + jn;
        tcol = "target_angle_" + jn;

        plot(t, T.(jcol), 'LineWidth', 1.3); hold on; grid on;
        if any(vars == tcol), plot(t, T.(tcol), '--', 'LineWidth', 1.1); end
        ylabel('angle [rad]');
        title(char(jn), 'Interpreter', 'none');
        lgd = {'joint\_angle', 'target\_angle'};
        if ~any(vars == tcol), lgd = {'joint\_angle'}; end
        legend(lgd, 'Location', 'best');
    end
end

% Figure 2: RPY ref vs simulated RPY (yaw_ref = 0)
has_attitude_cmd_new = all(ismember(["cmd_roll_ref", "cmd_pitch_ref"], vars));
has_attitude_cmd_old = all(ismember(["cmd_lin_vel_y", "cmd_ang_vel_z"], vars));
has_base_rpy = all(ismember(["base_roll", "base_pitch", "base_yaw"], vars));
if (has_attitude_cmd_new || has_attitude_cmd_old) && has_base_rpy
    if has_attitude_cmd_new
        roll_ref = T.("cmd_roll_ref");
        pitch_ref = T.("cmd_pitch_ref");
        roll_lbl = 'roll\_ref (cmd\_roll\_ref)';
        pitch_lbl = 'pitch\_ref (cmd\_pitch\_ref)';
    else
        roll_ref = T.("cmd_lin_vel_y");
        pitch_ref = T.("cmd_ang_vel_z");
        roll_lbl = 'roll\_ref (cmd\_lin\_vel\_y)';
        pitch_lbl = 'pitch\_ref (cmd\_ang\_vel\_z)';
    end
    yaw_ref = zeros(height(T), 1);
    roll_sim = T.("base_roll");
    pitch_sim = T.("base_pitch");
    yaw_sim = T.("base_yaw");

    f2 = figure('Name', 'RPY Ref vs Sim', 'Color', 'w');
    tiledlayout(f2, 3, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    plot(t, roll_ref, 'LineWidth', 1.4); hold on; grid on;
    plot(t, roll_sim, '--', 'LineWidth', 1.2);
    ylabel('roll [rad]');
    legend(roll_lbl, 'base\_roll', 'Location', 'best');
    title('Roll Tracking');

    nexttile;
    plot(t, pitch_ref, 'LineWidth', 1.4); hold on; grid on;
    plot(t, pitch_sim, '--', 'LineWidth', 1.2);
    ylabel('pitch [rad]');
    legend(pitch_lbl, 'base\_pitch', 'Location', 'best');
    title('Pitch Tracking');

    nexttile;
    plot(t, yaw_ref, 'LineWidth', 1.4); hold on; grid on;
    plot(t, yaw_sim, '--', 'LineWidth', 1.2);
    ylabel('yaw [rad]');
    xlabel('time');
    legend('yaw\_ref (0)', 'base\_yaw', 'Location', 'best');
    title('Yaw Tracking');
else
    warning('Attitude ref/sim columns not found. Need (cmd_roll_ref/cmd_pitch_ref) or (cmd_lin_vel_y/cmd_ang_vel_z), and base_roll, base_pitch, base_yaw.');
end

% Figure 3: Force Observer vs Sensor force (per leg, 4 subplots)
legs = ["FL", "FR", "RL", "RR"];
has_any_force_plot = false;
f3 = figure('Name', 'FOB vs Sensor Force (Per Leg)', 'Color', 'w');
tiledlayout(f3, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:numel(legs)
    leg = legs(i);
    fob_x = "FOB_force_body_x_" + leg;
    fob_y = "FOB_force_body_y_" + leg;
    fob_z = "FOB_force_body_z_" + leg;
    sen_x = "Sensor_force_body_x_" + leg;
    sen_y = "Sensor_force_body_y_" + leg;
    sen_z = "Sensor_force_body_z_" + leg;

    nexttile;
    if all(ismember([fob_x, fob_y, fob_z, sen_x, sen_y, sen_z], vars))
        fob_norm = sqrt(double(T.(fob_x)).^2 + double(T.(fob_y)).^2 + double(T.(fob_z)).^2);
        sen_norm = sqrt(double(T.(sen_x)).^2 + double(T.(sen_y)).^2 + double(T.(sen_z)).^2);
        plot(t, fob_norm, 'LineWidth', 1.3); hold on; grid on;
        plot(t, sen_norm, '--', 'LineWidth', 1.2);
        title(char(leg), 'Interpreter', 'none');
        ylabel('force norm');
        legend('FOB ||F||', 'Sensor ||F||', 'Location', 'best');
        has_any_force_plot = true;
    else
        axis off;
        text(0.5, 0.5, sprintf('%s columns missing', leg), 'HorizontalAlignment', 'center');
    end
end
xlabel('time');
if ~has_any_force_plot
    warning('FOB/Sensor force columns not found. Need FOB_force_body_[xyz]_* and Sensor_force_body_[xyz]_*.');
end

end
