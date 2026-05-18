function mark_resets(t_rl, reset_steps)
% mark_resets  Draw vertical green dashed lines at reset events.
    for k = 1:length(reset_steps)
        rs = reset_steps(k);
        if rs >= 1 && rs <= length(t_rl)
            xline(t_rl(rs), 'g--', 'reset', 'LabelVerticalAlignment', 'bottom');
        end
    end
end
