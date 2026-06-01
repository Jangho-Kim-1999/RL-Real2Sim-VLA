function run_estimation(csvPath, urdfPath)
% RUN_ESTIMATION  Quad CSV 로그 1 개에 대해
%                 [ InEKF + Taskspace FOB + GRF threshold contact ] 파이프라인을 돌리고, 
%                 body angular velocity 등을 reference command 와 비교해 plot 한다.
%
% 본 스크립트는 PerceptionCore::compute() (EST branch) 의 5 단계
%   1) compute_body          (FK in body frame)
%   2) StateEstimator::compute (InEKF predict + kinematic update)
%   3) sensor q/qd 의 base 부분을 SE 추정값으로 덮어쓴 RuntimeInput 합성
%   4) compute_world         (full FK + dynamics)
%   5) ForceEstimator + ContactEstimator
% 를 MATLAB 으로 그대로 옮겨 한 번에 돌린다.
%
% 함수 입력 :
%   csvPath  (기본 './data.csv')
%   urdfPath (기본 './Quad_v2_parallel.urdf')

  if nargin < 1 || isempty(csvPath)
    csvPath = fullfile(pwd, '260428_Dyna_Rand_Circle_0.5ms.csv');
  end
  if nargin < 2 || isempty(urdfPath)
    urdfPath = fullfile(pwd, 'Quad_v2_parallel.urdf');
  end

  fprintf('=== InEKF Body-State Estimation ===\n');
  fprintf('CSV  : %s\n', csvPath);
  fprintf('URDF : %s\n', urdfPath);

  % =========================================================================
  % STEP 0 : CONFIG (모든 튜닝 파라미터를 한 곳에)
  % =========================================================================
  cfg.dt                    = 0.001;       % 샘플 시간 (sec)

  % --- IMU 가속도 단위 ---------------------------------------------------
  %  'auto'  : 첫 1 s 의 |body_accel| 평균을 보고 1 (g) 또는 9.81 (m/s^2) 자동 판정
  %  'g'     : raw 가 g 단위 -> 9.81 곱해 m/s^2 로
  %  'm/s2'  : 이미 m/s^2 -> pass-through
  cfg.accel_unit            = 'auto';

  % --- InEKF Process, Measurement noise stds 설정 (PerceptionCoreTuning.inekf에서 설정된 default 값을 반영) ---
  cfg.gyro_std              = 0.005;       % rad/s
  cfg.accel_std             = 0.05;        % m/s^2
  cfg.gbias_std             = 1e-4;        % rad/s^2 random walk
  cfg.abias_std             = 1e-3;        % m/s^3   random walk
  cfg.contact_std           = 0.02;        % m       foot kinematic noise

  % --- Quadruped Robot의 초기 상태 / 공분산 설정 (정지 stance 가정) -----------------------------
  cfg.init.P_rpv            = 0.001;
  cfg.init.P_bias           = 1e-3;
  cfg.init.bg               = [0; 0; 0];
  cfg.init.ba               = [0; 0; 0];
  cfg.init.p                = [0; 0; 0.32];

  % --- GRF 추정을 위한 FOB의 tuning 값 설정 ---------------------------------------------------------
  cfg.fob_cutoff_hz         = 30;          % LPF cutoff (Hz, scalar -> 모든 leg/axis)
  cfg.fob_jac_sing_thresh   = 1e-4;

  % --- Contact 추정을 위한 GRF threshold contact estimator의 tuning 값 설정 -----------------------------------
  cfg.contact_thresh_N      = [10, 10, 10, 10];  % leg 별 [N]

  % --- Joint Position을 사용한 Joint Vel, Accel을 구하기 위한 수치 미분 LPF cutoff ----------------------------------------------
  cfg.diff_cutoff_hz        = 25;

  % --- Plot 마스킹 (warmup 동안의 transient 숨기기) ----------------------
  cfg.warmup_steps          = 50;

  % =========================================================================
  % STEP 1 : CSV 로드 + 컬럼 분리 + 가속도 단위 결정
  % =========================================================================
  fprintf('\n[1/5] Loading CSV...\n');
  T = readtable(csvPath, 'PreserveVariableNames', false);
  % 'PreserveVariableNames=false' 로 두면 컬럼명이 유효한 변수명으로 정규화됨.
  % 그래도 leading space 등이 남는 경우가 있어 strtrim 한 번 더.
  T.Properties.VariableNames = strtrim(T.Properties.VariableNames);
  N = height(T);
  fprintf('       %d samples  (%.3f s @ dt=%.4f s)\n', ...
          N, (N-1)*cfg.dt, cfg.dt);

  % --- 시계열 / 관절 / IMU / 명령 / reference 컬럼 추출 -------------------
  t            = T.t;

  % active joint 12 개 (FL HAA/HIP/DRIVER, FR..., RL..., RR...)
  joint_angle  = [T.joint_angle_0,  T.joint_angle_1,  T.joint_angle_2, ...
                  T.joint_angle_3,  T.joint_angle_4,  T.joint_angle_5, ...
                  T.joint_angle_6,  T.joint_angle_7,  T.joint_angle_8, ...
                  T.joint_angle_9,  T.joint_angle_10, T.joint_angle_11];   % Nx12

  % 모터 명령 토크 (FOB 의 tau_cmd 로 사용 — actuator_torque 대용)
  ctrl_input   = [T.ctrl_input_0,  T.ctrl_input_1,  T.ctrl_input_2, ...
                  T.ctrl_input_3,  T.ctrl_input_4,  T.ctrl_input_5, ...
                  T.ctrl_input_6,  T.ctrl_input_7,  T.ctrl_input_8, ...
                  T.ctrl_input_9,  T.ctrl_input_10, T.ctrl_input_11];      % Nx12
  
  % IMU 측정값
  body_rot_rate = [T.body_rot_rate_0, T.body_rot_rate_1, T.body_rot_rate_2]; % gyro [rad/s]
  body_accel_g  = [T.body_accel_0,    T.body_accel_1,    T.body_accel_2];    % accel [g 또는 m/s^2]

  % IMU 자체가 추정한 orientation angle (roll, pitch, yaw) [rad].
  body_rot      = [T.body_rot_0, T.body_rot_1, T.body_rot_2];              % Nx3

  % RL 명령 : (vx_cmd, vy_cmd, wz_cmd). vx, vy 는 figure 2 의 reference, wz 는 figure 4 의 wz subplot 에 overlay.
  vel_cmd       = [T.policy_velocity_command_0, ...
                   T.policy_velocity_command_1, ...
                   T.policy_velocity_command_2];                           % Nx3

  % --- 가속도 단위 결정 + m/s^2 로 통일 ----------------------------------
  switch lower(cfg.accel_unit)
    case 'auto'
      probeN = min(1000, N);
      mag    = mean(vecnorm(body_accel_g(1:probeN, :), 2, 2));
      if mag < 3.0           % 1 에 가까우면 g, 9.81 에 가까우면 m/s^2
        accel_scale = 9.81;
        unit_label  = 'g (auto)';
      else
        accel_scale = 1.0;
        unit_label  = 'm/s^2 (auto)';
      end
    case 'g'
      accel_scale = 9.81;
      unit_label  = 'g';
    case {'m/s2', 'm/s^2', 'mss'}
      accel_scale = 1.0;
      unit_label  = 'm/s^2';
    otherwise
      error('run_estimation:badAccelUnit', ...
            'cfg.accel_unit must be ''auto'', ''g'', or ''m/s2''.');
  end
  body_accel_ms2 = accel_scale * body_accel_g;
  fprintf('       accel unit detected: %s  (scale=%.4f)\n', unit_label, accel_scale);

  % =========================================================================
  % STEP 2 : 로봇 모델 로드 (URDF -> rigidBodyTree)
  % =========================================================================
  fprintf('\n[2/5] Loading URDF...\n');
  robot = load_urdf(urdfPath);
  jmap  = build_joint_map(robot);
  fprintf('       q_dim = %d (20 expected for fixed-base 5-bar quad)\n', ...
          jmap.q_dim);

  % =========================================================================
  % STEP 3 : 수치 미분으로 qd, qdd 추정 (CSV 에 직접 없으므로)
  %
  %   joint_angle [Nx12]
  %     -> Tustin LPF 미분 (cutoff = cfg.diff_cutoff_hz)
  %     -> joint_vel
  %     -> 다시 Tustin LPF 미분
  %     -> joint_acc
  % =========================================================================
  fprintf('\n[3/5] Numerical differentiation (joint vel/acc)...\n');
  joint_vel = lpf_diff_signal(joint_angle, cfg.dt, cfg.diff_cutoff_hz);
  joint_acc = lpf_diff_signal(joint_vel,   cfg.dt, cfg.diff_cutoff_hz);

  % =========================================================================
  % STEP 4 : 메인 루프 — InEKF + FOB + GRF threshold contact
  % =========================================================================
  fprintf('\n[4/5] Running estimation loop...\n');

  % 추정기 인스턴스 생성
  ekf = InvariantEKF(cfg.init);
  ekf.setNoise(cfg.gyro_std, cfg.accel_std, cfg.gbias_std, ...
               cfg.abias_std, cfg.contact_std);

  fob = TaskspaceFOB(cfg.dt, cfg.fob_cutoff_hz, cfg.fob_jac_sing_thresh);

  % 1-step 지연 contact feedback :
  %   InEKF 의 kinematic update 는 "이전 step 의 contact" 가 필요한데,
  %   contact estimator 는 InEKF 의 R 추정이 있어야 동작 (force 회전에 사용).
  %   순환을 끊기 위해 1-step 지연.
  last_contact = false(4, 1);

  % 출력 trace 어레이 사전 할당 -----------------------------------------
  est_omega_xyz = zeros(N, 3);    % bias 보정 angular velocity (body frame)
  est_bg        = zeros(N, 3);
  est_ba        = zeros(N, 3);
  est_rpy       = zeros(N, 3);    % roll-pitch-yaw
  est_p         = zeros(N, 3);
  est_v         = zeros(N, 3);    % world frame linear velocity (InEKF 직접 출력)
  contacts      = false(N, 4);
  fz_world      = zeros(N, 4);    % world frame 발 외력 z 성분

  % --- frame 변환 결과 저장용 (plot 시 사용) -----------------------------
  % policy_velocity_command 는 body frame 명령 (vx_body, vy_body, wz_body).
  % InEKF 의 v 는 world frame 이므로 비교하려면 body frame 으로 회전 필요.
  % gyro - bg 는 이미 body frame 이라 변환 불필요.
  est_v_body      = zeros(N, 3);  % body frame linear velocity = R^T * v_world

  tic;
  for k = 1 : N
    % -------- 이번 step 의 입력 ---------------------------------------
    q_active   = joint_angle(k, :)';     % 12x1
    qd_active  = joint_vel(k, :)';       % 12x1
    qdd_active = joint_acc(k, :)';       % 12x1
    gyro       = body_rot_rate(k, :)';   % 3x1
    accel      = body_accel_ms2(k, :)';  % 3x1

    % -------- STEP 4-1 : 평행사변형 FK 로 body-frame 발 위치/Jacobian --
    body          = compute_body_kinematics(robot, jmap, q_active);
    foot_pos_body = body.foot_pos_body;     % 3x4 (열이 한 leg)
    J_B           = body.foot_jac_body;     % 3x3x4
    q_full        = body.q_full;            % 20-dim, passive 채워진 q

    % -------- STEP 4-2 : InEKF predict + kinematic update --------------
    % propagate : strapdown + 공분산 전파
    ekf.propagate(gyro, accel, cfg.dt);

    % correctKinematics : last_contact 가 true 인 stance 발들의 측정값을
    %                     이용해 invariant residual 로 update.
    %                     발이 새로 stance 면 augment, swing 으로 빠지면 remove.
    ekf.correctKinematics(last_contact, foot_pos_body);

    R_est  = ekf.getR();
    bg_est = ekf.getBg();
    ba_est = ekf.getBa();

    % -------- STEP 4-3 : 정역학 (full RNEA, fixed-base) ----------------
    % FOB 가 필요로 하는 tau_dyn (각 active joint 에 작용하는 동역학 토크).
    %
    % MATLAB inverseDynamics(robot, q, qd, qdd) :
    %   tau = M(q) qdd + C(q,qd) qd + G(q)
    %
    % base 가 fixed 라 floating-base 의 base inertia/Coriolis 항은 누락
    % (정지 / 저속 데이터 한정 OK).
    qd_full  = active_to_full(qd_active,  jmap);
    qdd_full = active_to_full(qdd_active, jmap);

    tau_full = inverseDynamics(robot, q_full, qd_full, qdd_full);  % q_dim x 1

    % per-leg active joint 토크만 추출 (4x3) ----------------------------
    tau_dyn_4x3 = zeros(4, 3);
    tau_cmd_4x3 = zeros(4, 3);
    for L = 1 : 4
      tau_dyn_4x3(L, :) = tau_full(jmap.q_active_idx(L, :))';
      tau_cmd_4x3(L, :) = ctrl_input(k, 3*(L-1) + (1:3));    % CSV 의 ctrl_input
    end

    % -------- STEP 4-4 : Taskspace FOB -> body-frame 발 외력 ----------
    F_body_3x4 = fob.step(J_B, tau_dyn_4x3, tau_cmd_4x3);    % 3x4

    % -------- STEP 4-5 : world frame 으로 회전 -------------------------
    % GRF threshold 는 world Z 성분 (수직 외력) 으로 stance 판정.
    F_world_3x4   = R_est * F_body_3x4;       % 3x4
    fz_world(k, :) = F_world_3x4(3, :);

    % -------- STEP 4-6 : GRF threshold contact estimator ---------------
    % 원본 GRFThreshold.cc 의 :
    %     contact[leg] = sched_contact[leg] && (f_W.z > thresh)
    % 에서 sched_contact 게이팅 제거한 단순 GRF threshold.
    contact_now = false(4, 1);
    for L = 1 : 4
      contact_now(L) = F_world_3x4(3, L) > cfg.contact_thresh_N(L);
    end
    contacts(k, :) = contact_now';

    % --- 다음 step 의 InEKF kinematic update 를 위한 feedback ----------
    last_contact = contact_now;

    % -------- STEP 4-7 : 로깅 ------------------------------------------
    v_world_now = ekf.getV();           % world frame linear velocity (그대로)
    omega_body_now = gyro - bg_est;     % body frame angular velocity (raw - bg)

    est_omega_xyz(k, :) = omega_body_now';
    est_bg(k, :)        = bg_est';
    est_ba(k, :)        = ba_est';
    est_rpy(k, :)       = ekf.getRPY()';
    est_p(k, :)         = ekf.getP()';
    est_v(k, :)         = v_world_now';

    % --- frame 변환 (현재 step 의 R_est 로) -----------------------------
    % command 는 이미 body frame 이라 그대로 사용. v_world 만 body 로 회전.
    est_v_body(k, :) = (R_est' * v_world_now)';

    if mod(k, 1000) == 0
      fprintf('       step %6d / %d  (%.1f%%)  bg=[% .4f % .4f % .4f]\n', ...
              k, N, 100*k/N, bg_est(1), bg_est(2), bg_est(3));
    end
  end
  fprintf('       elapsed: %.2f s  (%.0f Hz effective)\n', toc, N/toc);

  % =========================================================================
  % STEP 5 : PLOT  (2 figures : Body Frame Velocity / Orientation)
  %
  % RL 명령은 body frame 컨벤션 (vx_body, vy_body, wz_body) 이므로 비교는
  % body frame 에서. world frame 으로 굳이 회전시키면 R 추정의 잡음이
  % reference 까지 흐릿하게 만들어 비교 의미가 흐려진다.
  %
  % Figure 1 의 6 subplot 구조 (3 row x 2 col) :
  %   row = axis (x, y, z),   좌 col = linear,   우 col = angular
  % =========================================================================
  fprintf('\n[5/5] Plotting...\n');

  % warmup transient 숨기기 위한 마스크
  m  = (1:N)' > cfg.warmup_steps;
  ts = t(m);

  % --- body frame command (학습 의도 그대로) ---------------------------
  v_cmd_body = [vel_cmd(:, 1), vel_cmd(:, 2), zeros(N, 1)];   % (vx_b, vy_b, vz_b=0)
  w_cmd_body = [zeros(N, 1), zeros(N, 1), vel_cmd(:, 3)];     % (wx_b=0, wy_b=0, wz_b)

  % =========================================================================
  % Figure 1 : Body Frame Velocity (command tracking)
  %   좌 col : 학습 명령 (검정 stairs) vs R^T * v_world (빨강)
  %   우 col : 학습 명령 (검정 stairs) vs gyro - b_g    (빨강)
  %            + raw gyro 점선 (b_g 효과 확인용)
  % =========================================================================
  fig1 = figure('Name', 'Body Frame Velocity (command tracking)', ...
                'Position', [80, 80, 1400, 900]);
  vL = {'v_x', 'v_y', 'v_z'};
  wL = {'\omega_x', '\omega_y', '\omega_z'};
  for ax = 1 : 3
    % --- 좌측 : linear ---------------------------------------------------
    subplot(3, 2, 2*(ax-1) + 1);
    stairs(ts, v_cmd_body(m, ax), 'k-', 'LineWidth', 1.4); hold on;
    plot  (ts, est_v_body(m, ax), 'r-', 'LineWidth', 1.0);
    grid on;
    ylabel(sprintf('%s [m/s]', vL{ax}));
    if ax == 1
      title('Body Frame : Linear Velocity');
      legend({'reference (training command)', ...
              'InEKF estimate (R^T \cdot v_{world})'}, 'Location', 'best');
    end
    if ax == 3, xlabel('time [s]'); end

    % --- 우측 : angular --------------------------------------------------
    subplot(3, 2, 2*ax);
    stairs(ts, w_cmd_body(m, ax),   'k-', 'LineWidth', 1.4); hold on; grid on;
    plot  (ts, est_omega_xyz(m, ax),'r-', 'LineWidth', 1.0);
    plot  (ts, body_rot_rate(m, ax),'b:', 'LineWidth', 0.7);
    ylabel(sprintf('%s [rad/s]', wL{ax}));
    if ax == 1
      title('Body Frame : Angular Velocity');
      legend({'reference (training command)', ...
              'InEKF estimate (gyro - b_g)', ...
              'raw gyro (body\_rot\_rate)'}, 'Location', 'best');
    end
    if ax == 3, xlabel('time [s]'); end
  end

  % =========================================================================
  % Figure 2 : Body Orientation (IMU vs InEKF RPY)
  %   roll, pitch : IMU 자체 추정과 InEKF 가 거의 같아야 정상
  %   yaw         : IMU 단독은 누적 drift, InEKF 는 발 접촉 landmark 로 anchor.
  %                 두 trace 가 갈라지면 InEKF 가 yaw 보정 중인 것.
  % =========================================================================
  fig2 = figure('Name', 'Body Orientation (IMU vs InEKF)', ...
                'Position', [120, 120, 1100, 720]);
  rpyL = {'roll', 'pitch', 'yaw'};
  for ax = 1 : 3
    subplot(3, 1, ax);
    plot(ts, body_rot(m, ax), 'k-', 'LineWidth', 1.2); hold on; grid on;
    plot(ts, est_rpy(m, ax),  'r-', 'LineWidth', 1.0);
    ylabel(sprintf('%s [rad]', rpyL{ax}));
    if ax == 1
      title('Body Orientation (IMU self-estimate vs InEKF)');
      legend({'IMU (body\_rot)', 'InEKF estimate'}, 'Location', 'best');
    end
    if ax == 3, xlabel('time [s]'); end
  end

  % =========================================================================
  % 결과 export : base workspace 로 inekf_results 구조체 전달
  % =========================================================================
  results.t                = t;
  results.est_omega_xyz    = est_omega_xyz;        % body frame angular vel (gyro - bg)
  results.est_bg           = est_bg;
  results.est_ba           = est_ba;
  results.est_rpy          = est_rpy;              % InEKF RPY (ZYX Euler)
  results.est_p            = est_p;
  results.est_v            = est_v;                % world frame linear vel (InEKF 직접)
  results.est_v_body       = est_v_body;           % body frame linear vel  (R^T * v_world)
  results.contacts         = contacts;
  results.fz_world         = fz_world;
  results.body_accel_ms2   = body_accel_ms2;
  results.body_rot         = body_rot;             % IMU self-estimate RPY (CSV 직접)
  results.vel_cmd          = vel_cmd;              % body frame command (CSV 직접)
  results.cfg              = cfg;
  results.unit_label       = unit_label;
  assignin('base', 'inekf_results', results);
  fprintf('\nDone. Results placed in base workspace as `inekf_results`.\n');
  fprintf('Figures: 1=Body Frame Velocity, 2=Orientation.\n');
end

% =========================================================================
% 로컬 helper : Tustin LPF 미분 (filter.cc::tustin_derivative 와 동일 식)
%
%   y_k = (2*(x_k - x_{k-1}) - (Ts - 2 tau_c) y_{k-1}) / (Ts + 2 tau_c)
%   tau_c = 1 / (2 pi fc)
%
% 한 column 씩 처리. 첫 step 에는 x_old = x_0 으로 둬서 순간 변화 없음.
% =========================================================================
function out = lpf_diff_signal(x, dt, cutoff_hz)
  [N, D] = size(x);
  out    = zeros(N, D);
  wc     = 2 * pi * cutoff_hz;
  tc     = 1.0 / wc;
  for j = 1 : D
    in     = x(:, j);
    in_old = [in(1); in(1:end-1)];
    out_col = zeros(N, 1);
    for k = 2 : N
      out_col(k) = (2*(in(k) - in_old(k)) - (dt - 2*tc) * out_col(k-1)) ...
                                          / (dt + 2*tc);
    end
    out(:, j) = out_col;
  end
end

% =========================================================================
% 로컬 helper : 12 active 값을 평행사변형 closure 로 q_dim full vector 로 확장.
%
% 같은 사상 행렬이 q, qd, qdd 에 모두 적용된다 (선형 제약을 미분하면 같음).
%
%   knee     = driver - hip
%   coupling = -knee  = hip - driver
% =========================================================================
function q_full = active_to_full(q_active_12, jmap)
  q_full = zeros(jmap.q_dim, 1);
  for L = 1 : 4
    haa    = q_active_12(3*(L-1) + 1);
    hip    = q_active_12(3*(L-1) + 2);
    drv    = q_active_12(3*(L-1) + 3);
    knee   = drv - hip;
    coupl  = -knee;
    q_full(jmap.q_active_idx(L, 1))  = haa;
    q_full(jmap.q_active_idx(L, 2))  = hip;
    q_full(jmap.q_active_idx(L, 3))  = drv;
    q_full(jmap.q_passive_idx(L, 1)) = knee;
    q_full(jmap.q_passive_idx(L, 2)) = coupl;
  end
end
