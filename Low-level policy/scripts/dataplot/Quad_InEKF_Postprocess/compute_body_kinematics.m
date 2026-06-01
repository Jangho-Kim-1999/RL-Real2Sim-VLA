function out = compute_body_kinematics(robot, jmap, q_active_12)
% COMPUTE_BODY_KINEMATICS  4 다리의 body-frame 발 위치 / Jacobian 계산.
%
% C++ 측 robot_parameter::compute_body_only() 의 MATLAB 대응이지만, 본
% 로봇의 biarticular 5-bar leg 의 평행사변형 폐쇄 조건을 닫힌 형태로
% 적용해 active 12 개 joint 만으로 passive 8 개를 얻는다 (수치 솔버 불요).
%
% 인자 :
%   robot       : setup_robot 결과 rigidBodyTree
%   jmap        : build_joint_map(robot) 결과 매핑
%   q_active_12 : 12x1 active joint 각도, 순서 = leg 별 [HAA;HIP;DRIVER]
%                 = [FL_HAA;FL_HIP;FL_DRIVER;
%                    FR_HAA;FR_HIP;FR_DRIVER;
%                    RL_HAA;RL_HIP;RL_DRIVER;
%                    RR_HAA;RR_HIP;RR_DRIVER]
%
% 반환 struct out :
%   out.q_full         : (q_dim x 1)  passive 까지 채워진 full q-vector
%   out.foot_pos_body  : 3x4    leg 별 IMU frame 발 위치 (각 열이 한 leg)
%   out.foot_jac_body  : 3x3x4  leg 별 IMU frame 발 Jacobian (active 3-DOF)
%
% --- 평행사변형 폐쇄 조건 (URDF 분석 결과 도출) ---
% URDF 측정으로 다음을 확인 :
%   thigh 길이      = coupler 길이      = 0.25 m
%   knee->loop_pivot = driver_link 길이 = 0.06 m
%   HIP joint 와 DRIVER joint 가 동축 (origin 동일)
% 따라서 진정한 4-bar 평행사변형이며 폐쇄 조건이 다음과 같이 단순화된다 :
%
%     theta_KNEE      = theta_DRIVER - theta_HIP
%     theta_COUPLING  = theta_HIP   - theta_DRIVER  =  -theta_KNEE
%
% 이를 미분하면 Jacobian projection 이 상수 행렬로 떨어진다 :
%
%   [ KNEE_dot ]   [ 0  -1   1 ] [ HAA_dot     ]
%   [          ] = [           ] [ HIP_dot     ]
%   [ COUPL_dot]   [ 0   1  -1 ] [ DRIVER_dot  ]
%
% 따라서 C++ 의 pinv(J_constraint_passive) * J_constraint_active 단계가
% 닫힌 형태 행렬 Gamma 로 대체된다.

  % =========================================================================
  % STEP 1 : 평행사변형으로 passive 8 개를 채워 full q (20-dim) 구성
  % =========================================================================
  q = zeros(jmap.q_dim, 1);
  for leg = 1 : 4
    haa     = q_active_12(3*(leg-1) + 1);
    hip     = q_active_12(3*(leg-1) + 2);
    driver  = q_active_12(3*(leg-1) + 3);

    % 평행사변형 폐쇄 조건 적용
    knee    = driver - hip;
    coupl   = -knee;          % == hip - driver

    q(jmap.q_active_idx(leg, 1))  = haa;
    q(jmap.q_active_idx(leg, 2))  = hip;
    q(jmap.q_active_idx(leg, 3))  = driver;
    q(jmap.q_passive_idx(leg, 1)) = knee;
    q(jmap.q_passive_idx(leg, 2)) = coupl;
  end

  out.q_full = q;

  % =========================================================================
  % STEP 2 : leg 별로 발 위치 / Jacobian (IMU frame) 산출
  % =========================================================================
  out.foot_pos_body = zeros(3, 4);
  out.foot_jac_body = zeros(3, 3, 4);

  % 평행사변형 폐쇄 조건의 미분형 (상수). 행 = [KNEE; COUPLING], 열 = [HAA, HIP, DRIVER].
  Gamma = [0 -1 +1;
           0 +1 -1];

  for leg = 1 : 4
    foot_name = jmap.foot_frame{leg};

    % --- (a) IMU frame 에서 본 발 위치 -------------------------------------
    % T_imu^foot = (imu_link 의 world pose)^-1 * (foot 의 world pose)
    % MATLAB getTransform(robot, q, target, source) 가 그대로 변환행렬을 줌.
    T_body_to_foot = getTransform(robot, q, foot_name, jmap.imu_frame);
    out.foot_pos_body(:, leg) = T_body_to_foot(1:3, 4);

    % --- (b) 발 frame 의 Jacobian (base 기준, 6 x q_dim) -------------------
    % geometricJacobian 은 LOCAL_WORLD_ALIGNED 와 비슷한 의미 :
    %   행 1..3 = angular, 행 4..6 = linear, 좌표는 base frame.
    % imu_link 가 base_link 와 같은 좌표축이므로 base frame == IMU frame.
    Jfull = geometricJacobian(robot, q, foot_name);
    Jlin  = Jfull(4:6, :);                  % 3 x q_dim, body frame 의 linear 부분

    % --- (c) 이 leg 의 5 개 column 만 추출 (active 3 + passive 2) ---------
    iAct = jmap.q_active_idx(leg, :);       % 1x3 (HAA, HIP, DRIVER)
    iPas = jmap.q_passive_idx(leg, :);      % 1x2 (KNEE, COUPLING)

    Jact = Jlin(:, iAct);                   % 3x3
    Jpas = Jlin(:, iPas);                   % 3x2

    % --- (d) 평행사변형 closure 적용한 effective Jacobian -----------------
    % dq_passive = Gamma * dq_active 이므로
    %   dx = Jact * dq_active + Jpas * dq_passive
    %      = (Jact + Jpas * Gamma) * dq_active
    Jeff = Jact + Jpas * Gamma;             % 3x3

    out.foot_jac_body(:, :, leg) = Jeff;
  end
end
