classdef InvariantEKF < handle
% INVARIANTEKF  Right-Invariant EKF (Hartley 2018) 의 MATLAB 포팅.
%
% 원본 :
%   Biarticular_Quad_Ctrl/robot/Perception/StateEstimator/InEKF.cc
%   (Hartley et al., "Contact-Aided Invariant EKF for Legged Robot State
%    Estimation", RSS 2018, arXiv:1805.10410 의 lean re-impl)
%
% 본 클래스는 위 C++ Impl 의 propagate / correct / correctKinematics 를
% 1:1 로 옮긴 것. SE_K(3) 의 K (= 현재 stance 인 발 수) 는 매 step 동적
% 으로 변하며, 이를 contact_slot_ 으로 추적한다.
%
% =========================================================================
% State 표현 (SE_K(3))
% =========================================================================
%
%   X_  :  (3+K) x (3+K)  상태 행렬 (Lie 군 위의 원소)
%
%        column 1..3 : R   (회전, 3x3)
%        column 4    : v   (속도, world)
%        column 5    : p   (위치, world)
%        column 6..  : d_i (stance 발 i 의 world 위치, K 개)
%
%        (4,4) 부터 (3+K, 3+K) 까지의 대각 = 1, 그 외 0  (SE_K(3) 형식)
%
%   Theta_  :  6 x 1  =  [bias_gyro (3) ; bias_accel (3)]
%
%   P_  :  (15 + 3K) x (15 + 3K)  공분산
%
%        rows 1..3   : phi   (회전 오차)
%        rows 4..6   : v
%        rows 7..9   : p
%        rows 10..12,13..15 : 첫번째 landmark 가 있다면 그 d_1 (3차원),
%                              없다면 bias_gyro 와 bias_accel
%        ... (landmark 들이 차례로 추가/제거되면서 P 의 차원이 변함)
%        rows end-5..end-3 : bias_gyro
%        rows end-2..end   : bias_accel
%
%   contact_slot_(leg) : leg (1..4) 가 현재 X_ 의 몇 번째 column 에 들어
%                        있는지를 저장. 1-indexed (=>  >= 6).
%                        leg 가 swing 이면 -1.
%
% =========================================================================
% 사용법 (한 step 당) :
%   ekf.propagate(gyro, accel, dt)
%   ekf.correctKinematics(contact_prev, foot_pos_body)
%
% Tunable noise stds 는 setNoise(...) 로 언제든 변경 가능.
% =========================================================================

  properties (Access = private)
    X_                % (3+K) x (3+K)
    Theta_            % 6x1
    P_                % (15+3K) x (15+3K)

    contact_slot_     % 4x1, 각 leg 의 X_ 내 1-indexed column 번호 (-1 = swing)

    % Noise 공분산 블록 (3x3 diagonal). setNoise() 로 갱신.
    Qg_; Qa_; Qbg_; Qba_; Qc_;
  end

  properties (Constant)
    kGravity   = [0; 0; -9.81];
    kNumFeet   = 4;
    kDimTheta  = 6;
  end

  % =======================================================================
  % public methods
  % =======================================================================
  methods
    % -----------------------------------------------------------------
    % 생성자 :  C++ 측 InEKF::Impl::Impl() 의 하드코딩 초기화에 대응.
    %
    % opts (struct, 모든 필드 선택) :
    %   .R, .v, .p   초기 자세 (기본 I, 0, [0;0;0.35])
    %   .bg, .ba     초기 bias  (기본 0)
    %   .P_rpv       R/v/p 블록 초기 분산 (기본 0.01)
    %   .P_bias      bias 블록   초기 분산 (기본 1e-4)
    % -----------------------------------------------------------------
    function obj = InvariantEKF(opts)
      if nargin < 1, opts = struct(); end

      R0   = getOrDefault(opts, 'R',  eye(3));
      v0   = getOrDefault(opts, 'v',  zeros(3,1));
      p0   = getOrDefault(opts, 'p',  [0;0;0.35]);
      bg0  = getOrDefault(opts, 'bg', zeros(3,1));
      ba0  = getOrDefault(opts, 'ba', zeros(3,1));
      sRvp = getOrDefault(opts, 'P_rpv',  0.01);
      sBs  = getOrDefault(opts, 'P_bias', 1e-4);

      % --- X_ : 5x5 단위행렬에서 시작 (K=0). 이후 column R / v / p 채움 ---
      obj.X_ = eye(5);
      obj.X_(1:3, 1:3) = R0;
      obj.X_(1:3, 4)   = v0;
      obj.X_(1:3, 5)   = p0;

      % --- Theta_ : bias 들 ---
      obj.Theta_ = [bg0; ba0];

      % --- P_ : 15x15 (K=0). diag 블록별로 분산 다르게 ---
      obj.P_ = eye(15);
      obj.P_(1:3,   1:3)   = sRvp * eye(3);   % 회전 phi
      obj.P_(4:6,   4:6)   = sRvp * eye(3);   % 속도 v
      obj.P_(7:9,   7:9)   = sRvp * eye(3);   % 위치 p
      obj.P_(10:12, 10:12) = sBs  * eye(3);   % bias_gyro
      obj.P_(13:15, 13:15) = sBs  * eye(3);   % bias_accel

      % --- 시작 시점에는 stance 발 없음 ---
      obj.contact_slot_ = [-1; -1; -1; -1];

      % 합리적인 default noise. 실제 사용자는 setNoise() 로 갱신할 것.
      obj.setNoise(0.01, 0.10, 1e-5, 1e-4, 0.01);
    end

    % -----------------------------------------------------------------
    % setNoise : noise 표준편차 -> 공분산 블록 (Q = sigma^2 I)
    %   gyro_std    : rad/s/sqrt(s)  (white)
    %   accel_std   : m/s^2/sqrt(s)
    %   gbias_std   : rad/s^2/sqrt(s) (random walk)
    %   abias_std   : m/s^3/sqrt(s)   (random walk)
    %   contact_std : m/sqrt(s)       (foot slip / kinematic 측정 잡음)
    % -----------------------------------------------------------------
    function setNoise(obj, gyro_std, accel_std, gbias_std, abias_std, contact_std)
      obj.Qg_  = (gyro_std    ^2) * eye(3);
      obj.Qa_  = (accel_std   ^2) * eye(3);
      obj.Qbg_ = (gbias_std   ^2) * eye(3);
      obj.Qba_ = (abias_std   ^2) * eye(3);
      obj.Qc_  = (contact_std ^2) * eye(3);
    end

    % =================================================================
    % propagate : IMU 한 step 적분 + 공분산 전파
    % (C++ 측 Impl::propagate() 와 동일)
    %
    % 핵심 식 :
    %   w = gyro  - bg
    %   a = accel - ba
    %   R_pred = R * Exp_SO3(w * dt)
    %   v_pred = v + (R*a + g_W) * dt
    %   p_pred = p + v*dt + 0.5*(R*a + g_W)*dt^2
    %
    %   P_new = Phi * P * Phi'  +  Phi * Adj * Q * Adj' * Phi' * dt
    %     where  A    = invariant error dynamics 의 Jacobian
    %            Phi  = I + A*dt   (1차 근사)
    %            Adj  = blkdiag(Adjoint_SEK3(X), I_6)
    % =================================================================
    function propagate(obj, gyro, accel, dt)
      % -------- bias 보정 -------------------------------------------------
      bg = obj.Theta_(1:3);
      ba = obj.Theta_(4:6);
      w  = gyro  - bg;
      a  = accel - ba;

      % 현재 상태 스냅샷
      X = obj.X_;
      P = obj.P_;
      R = X(1:3, 1:3);
      v = X(1:3, 4);
      p = X(1:3, 5);

      % =====================================================================
      % STEP A : strapdown 운동 모델 (X 갱신)
      % =====================================================================
      phi    = w * dt;
      R_pred = R * LieGroup.ExpSO3(phi);
      v_pred = v + (R*a + obj.kGravity) * dt;
      p_pred = p + v*dt + 0.5*(R*a + obj.kGravity) * dt^2;

      obj.X_(1:3, 1:3) = R_pred;
      obj.X_(1:3, 4)   = v_pred;
      obj.X_(1:3, 5)   = p_pred;
      % 주: landmark column 들 (column 6..) 은 propagate 단계에서는 변하지 않음
      % (right-invariant error frame 에서 정지 가정).

      % =====================================================================
      % STEP B : invariant error dynamics 의 선형화 행렬 A 구성
      %
      % 기본 inertial term :
      %   A[v 행 block, phi 열 block] = [g_W]×
      %   A[p 행 block, v   열 block] = I
      %
      % bias coupling term :
      %   A[phi 행, bg 열]            = -R
      %   A[v   행, ba 열]            = -R
      %   A[v / p / d_i 행 block, bg 열] = -[X(:, j)]× R   (j = v(=4), p(=5), 또는 landmark)
      % =====================================================================
      dimX     = size(obj.X_, 1);
      dimP     = size(obj.P_, 1);
      dimTheta = obj.kDimTheta;
      thetaStart = dimP - dimTheta;       % theta 블록 시작 직전 인덱스 (0-base 끝)

      A = zeros(dimP, dimP);

      % 기본 inertial 항
      A(4:6, 1:3) = LieGroup.skew(obj.kGravity);   % d(v) / d(phi)
      A(7:9, 4:6) = eye(3);                        % d(p) / d(v)

      % bias coupling : phi -> bg, v -> ba
      A(1:3,  thetaStart+1 : thetaStart+3) = -R;
      A(4:6,  thetaStart+4 : thetaStart+6) = -R;

      % v / p / 모든 landmark 의 row block 에서 bg 와의 coupling
      %   X 의 column j (j=4: v, j=5: p, j>=6: landmark) 에 대해
      %   해당 row block 의 시작 위치 = 3*j - 8 (1-indexed)
      %   (검증 :  j=4 -> 4 ;  j=5 -> 7 ;  j=6 -> 10 ;  j=7 -> 13)
      for j = 4 : dimX
        d_j = X(1:3, j);
        rs  = 3*j - 8;        % row block 시작 (1-indexed)
        A(rs : rs+2, thetaStart+1 : thetaStart+3) = -LieGroup.skew(d_j) * R;
      end

      % =====================================================================
      % STEP C : process noise Q 의 블록 구성
      %   Q = blkdiag( Qg, Qa, Qc * (각 stance landmark), Qbg, Qba )
      % =====================================================================
      Qk = zeros(dimP, dimP);
      Qk(1:3, 1:3) = obj.Qg_;      % gyro white noise
      Qk(4:6, 4:6) = obj.Qa_;      % accel white noise

      % 각 stance leg 의 landmark 행/열 블록에 Qc 주입.
      for leg = 1 : obj.kNumFeet
        slot = obj.contact_slot_(leg);
        if slot < 0, continue; end
        rs = 3*slot - 8;           % 위 j 의 식과 동일한 행 블록 시작
        Qk(rs : rs+2, rs : rs+2) = obj.Qc_;
      end

      % bias 들의 random-walk noise
      Qk(thetaStart+1 : thetaStart+3, thetaStart+1 : thetaStart+3) = obj.Qbg_;
      Qk(thetaStart+4 : thetaStart+6, thetaStart+4 : thetaStart+6) = obj.Qba_;

      % =====================================================================
      % STEP D : 이산화 (1차) + 공분산 전파
      %
      %   Phi    = I + A*dt
      %   Adj    = blkdiag(Adjoint_SEK3(X), I_6)
      %   Q_hat  = Phi * Adj * Q * Adj' * Phi' * dt
      %   P_new  = Phi * P * Phi' + Q_hat
      % =====================================================================
      I   = eye(dimP);
      Phi = I + A * dt;

      Adj = I;
      Adj(1:thetaStart, 1:thetaStart) = LieGroup.AdjointSEK3(obj.X_);

      PhiAdj = Phi * Adj;
      Qk_hat = PhiAdj * Qk * PhiAdj' * dt;

      obj.P_ = Phi * P * Phi' + Qk_hat;
    end

    % =================================================================
    % correctKinematics : stance 발 kinematic 측정 update + slot 관리
    % (C++ 측 Impl::correct_kinematics() 와 동일한 4 phase)
    %
    % 인자 :
    %   contact_prev   : 4x1 logical (이전 step 의 contact 상태)
    %   foot_pos_body  : 3x4 (각 열 = leg 의 body-frame 발 위치 p_bc)
    %
    % 동작 흐름 :
    %   Phase 1 - 분류 : (contact_prev, slot 존재 여부) 조합으로
    %                     update / augment / remove / skip 결정
    %   Phase 2 - update : 누적된 측정값으로 stacked observation 만들고 correct()
    %   Phase 3 - remove : swing 으로 전환된 발들의 column 을 X 에서 제거
    %   Phase 4 - augment: 새로 stance 가 된 발을 X 에 새 column 으로 추가
    % =================================================================
    function correctKinematics(obj, contact_prev, foot_pos_body)
      R = obj.X_(1:3, 1:3);

      % =================================================================
      % PHASE 1 : leg 별 분류
      % =================================================================
      updates      = struct('leg', {}, 'slot', {}, 'pos', {}, 'cov', {});
      remove_legs  = [];
      augment_legs = [];

      for leg = 1 : obj.kNumFeet
        c     = logical(contact_prev(leg));
        slot  = obj.contact_slot_(leg);
        found = (slot >= 0);

        if ~c && found
          remove_legs(end+1) = leg; %#ok<AGROW>     % stance->swing : 제거
        elseif c && ~found
          augment_legs(end+1) = leg; %#ok<AGROW>    % swing->stance : 추가
        elseif c && found
          % 계속 stance : 일반 update 대상
          k = numel(updates) + 1;
          updates(k).leg  = leg;
          updates(k).slot = slot;
          updates(k).pos  = foot_pos_body(:, leg);
          updates(k).cov  = obj.Qc_;
        end
        % (~c && ~found) : 계속 swing -> skip
      end

      % =================================================================
      % PHASE 2 : 일반 update (stacked measurement)
      %
      % 각 measurement k 는 SE_K(3) invariant 잔차 형태 :
      %   y_k = [ p_bc ; 0 ; 1 (at p column = 5) ; 0... ; -1 (at d_slot col); 0...]
      %   b_k = [ 0    ; 0 ; 1 (at p column)     ; 0... ; -1 (at d_slot col); 0...]
      %
      % H_k 의 행 (3 행) :
      %   p 블록 (P 의 col 7..9) 에 -I, d_slot 블록에 +I
      %
      % N_k = R * Qc * R'   (body-frame 잡음을 world 로 회전)
      % =================================================================
      if ~isempty(updates)
        dimX = size(obj.X_, 1);
        dimP = size(obj.P_, 1);
        nm   = numel(updates);

        Y  = zeros(nm * dimX, 1);
        b  = zeros(nm * dimX, 1);
        H  = zeros(3*nm, dimP);
        N  = zeros(3*nm, 3*nm);
        PI = zeros(3*nm, nm * dimX);    % 각 측정의 평행이동 3 성분만 추출

        for k = 1 : nm
          u    = updates(k);
          Yoff = (k-1) * dimX;     % 이번 측정의 Y 블록 시작 오프셋 (0-base)

          % --- Y / b 채우기 ----------------------------------------------
          % Y(1:3) = body-frame 발 위치
          Y(Yoff+1 : Yoff+3) = u.pos;
          % p column index = 5  (X 에서 p 가 5번째 열, 1-indexed)
          Y(Yoff + 5)        = 1.0;
          Y(Yoff + u.slot)   = -1.0;

          b(Yoff + 5)        = 1.0;
          b(Yoff + u.slot)   = -1.0;

          % --- H 채우기 (P 공간의 선형화) -------------------------------
          % p 블록 (P col 7..9) 에 -I
          H(3*(k-1)+1 : 3*k, 7:9) = -eye(3);
          % d_slot 의 P 행 블록 시작 = 3*slot - 8 (1-indexed)
          dSlotStart = 3*u.slot - 8;
          H(3*(k-1)+1 : 3*k, dSlotStart : dSlotStart+2) = eye(3);

          % --- N : body-frame 잡음을 world frame 으로 ---------------------
          N(3*(k-1)+1 : 3*k, 3*(k-1)+1 : 3*k) = R * u.cov * R';

          % --- PI : 측정 k 의 평행이동 3 성분 추출 ------------------------
          PI(3*(k-1)+1 : 3*k, Yoff+1 : Yoff+3) = eye(3);
        end

        % stacked observation 을 correct() 로 처리 (right-invariant update)
        obj.correct(Y, b, H, N, PI);
      end

      % =================================================================
      % PHASE 3 : 제거
      % swing 으로 빠진 leg 의 column 을 X 에서 제거하고, 대응 P 행/열도 제거.
      % 큰 slot 부터 처리해야 다른 leg 의 slot index 가 어긋나지 않는다.
      % =================================================================
      if ~isempty(remove_legs)
        slots = arrayfun(@(l) obj.contact_slot_(l), remove_legs);
        [~, ord] = sort(slots, 'descend');
        remove_legs = remove_legs(ord);

        for leg = remove_legs
          slot = obj.contact_slot_(leg);

          % --- X 에서 해당 column / row 제거 -----------------------------
          obj.X_(slot, :) = [];
          obj.X_(:, slot) = [];

          % --- P 에서 해당 3 행/열 제거 -----------------------------------
          pBlock = 3*slot - 8;        % 1-indexed P 블록 시작
          obj.P_(pBlock : pBlock+2, :) = [];
          obj.P_(:, pBlock : pBlock+2) = [];

          % --- contact_slot 갱신 + 더 큰 slot 들 한 칸씩 당김 ------------
          obj.contact_slot_(leg) = -1;
          for other = 1 : obj.kNumFeet
            if obj.contact_slot_(other) > slot
              obj.contact_slot_(other) = obj.contact_slot_(other) - 1;
            end
          end
        end
      end

      % =================================================================
      % PHASE 4 : 추가
      % 새로 stance 가 된 leg : foot 의 world 위치를 새 landmark column 으로
      % 추가. 새 d 의 초기값 = 현재 추정 p + R * p_bc.
      %
      % 공분산 확장 :
      %   F = [ I_core_old ;  e6 (p column) ; I_theta ]
      %   G = [ 0 ; ... ; R ; 0 ]    (새 행 위치에만 R)
      %   P_new = F * P_old * F' + G * Qc * G'
      % =================================================================
      if ~isempty(augment_legs)
        for leg = augment_legs
          X_old      = obj.X_;
          P_old      = obj.P_;
          dimX_old   = size(X_old, 1);
          dimP_old   = size(P_old, 1);
          dimTheta   = obj.kDimTheta;
          dimP_core  = dimP_old - dimTheta;       % theta 블록 직전까지의 길이

          R_now = X_old(1:3, 1:3);
          p_now = X_old(1:3, 5);

          % 새 landmark : world 좌표 = p + R * p_bc
          d_new = p_now + R_now * foot_pos_body(:, leg);

          % --- X 한 칸 키우기 + 새 column 에 d_new 채우기 ----------------
          X_new                          = eye(dimX_old + 1);
          X_new(1:dimX_old, 1:dimX_old)  = X_old;
          X_new(1:3, dimX_old + 1)       = d_new;

          % --- F : (dimP_old + 3) x dimP_old --------------------------
          %   기존 core 그대로 carry, 새 landmark 행 블록에 p column (col 7..9) 복사,
          %   theta 블록도 그대로 carry.
          F = zeros(dimP_old + 3, dimP_old);
          F(1 : dimP_core, 1 : dimP_core)                                   = eye(dimP_core);
          F(dimP_core+1 : dimP_core+3, 7:9)                                 = eye(3);
          F(dimP_core+4 : dimP_core+3+dimTheta, dimP_core+1 : dimP_core+dimTheta) = eye(dimTheta);

          % --- G : 새 landmark 행에 R 주입 (외부 잡음 입력 포지션) ------
          G = zeros(dimP_old + 3, 3);
          G(dimP_core+1 : dimP_core+3, :) = R_now;

          % --- 공분산 확장 -----------------------------------------------
          P_new = F * P_old * F' + G * obj.Qc_ * G';

          obj.X_ = X_new;
          obj.P_ = P_new;
          % 새 column 의 1-indexed 위치 = dimX_old + 1
          obj.contact_slot_(leg) = dimX_old + 1;
        end
      end
    end

    % ============================== accessors =========================
    function R  = getR(obj),  R  = obj.X_(1:3, 1:3); end
    function v  = getV(obj),  v  = obj.X_(1:3, 4);   end
    function p  = getP(obj),  p  = obj.X_(1:3, 5);   end
    function bg = getBg(obj), bg = obj.Theta_(1:3);  end
    function ba = getBa(obj), ba = obj.Theta_(4:6);  end
    function X  = getX(obj),  X  = obj.X_;           end
    function P  = getP_full(obj), P = obj.P_;        end
    function s  = getSlots(obj),  s = obj.contact_slot_; end

    % R -> intrinsic ZYX (yaw-pitch-roll) 변환 후 [roll; pitch; yaw] 반환
    function rpy = getRPY(obj)
      R = obj.getR();
      rpy = [atan2(R(3,2), R(3,3));
             atan2(-R(3,1), sqrt(R(3,2)^2 + R(3,3)^2));
             atan2(R(2,1), R(1,1))];
    end
  end

  % =======================================================================
  % private methods : right-invariant correction
  % =======================================================================
  methods (Access = private)

    % =================================================================
    % correct : right-invariant 측정 update.
    % (C++ Impl::correct() 1:1 포팅)
    %
    % 입력 :
    %   Y, b, H, N, PI : Phase 2 에서 만든 stacked observation 행렬들
    %
    % 핵심 식 :
    %   K     = P H' (H P H' + N)^{-1}
    %   BigX  = blkdiag(X, X, ..., X)        ← 측정 개수만큼 복제
    %   Z     = BigX * Y - b                 ← invariant residual
    %   delta = K * PI * Z
    %     delta 의 앞부분 = SE_K(3) error,  뒤 6 = bias error
    %   X     <- ExpSEK3(delta_X) * X        ← right-invariant : left-mult
    %   Theta <- Theta + delta_Theta
    %   P     <- (I - KH) P (I - KH)' + K N K'   (Joseph form)
    % =================================================================
    function correct(obj, Y, b, H, N, PI)
      P = obj.P_;

      % --- Kalman gain ---------------------------------------------------
      PHT = P * H';
      S   = H * PHT + N;
      K   = PHT / S;            % == PHT * inv(S)

      % --- BigX = X 를 측정 개수만큼 block-diag 복제 --------------------
      dimX  = size(obj.X_, 1);
      n_meas = numel(Y) / dimX;
      BigX  = kron(eye(n_meas), obj.X_);

      % --- 잔차 Z 와 update vector delta --------------------------------
      Z     = BigX * Y - b;
      delta = K * PI * Z;

      % delta 분리 : 앞 = SE_K(3) error, 뒤 = bias (6-dim)
      dimP     = size(obj.P_, 1);
      dimTheta = obj.kDimTheta;
      dXvec    = delta(1 : dimP - dimTheta);
      dTheta   = delta(dimP - dimTheta + 1 : end);

      % SE_K(3) error 를 group element 로
      dX = LieGroup.ExpSEK3(dXvec);

      % --- right-invariant update ---------------------------------------
      obj.X_     = dX * obj.X_;
      obj.Theta_ = obj.Theta_ + dTheta;

      % --- 공분산 update (Joseph form, 수치 안정) ---------------------
      I   = eye(dimP);
      IKH = I - K * H;
      obj.P_ = IKH * P * IKH' + K * N * K';
    end

  end
end

% =================================================================
% 로컬 유틸 : struct 의 default 값 추출
% =================================================================
function val = getOrDefault(s, field, default)
  if isfield(s, field) && ~isempty(s.(field))
    val = s.(field);
  else
    val = default;
  end
end
