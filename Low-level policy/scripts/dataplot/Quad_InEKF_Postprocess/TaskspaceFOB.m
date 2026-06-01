classdef TaskspaceFOB < handle
% TASKSPACEFOB  Force Observer 기반 leg 별 발 외력 추정.
%
% Biarticular_Quad_Ctrl/robot/Perception/ForceEstimator/TaskspaceFOB.cc 의
% MATLAB 포팅. C++ 와 신호 흐름 / LPF 식이 동일.
%
% 알고리즘 (leg 별, 축 별) :
%
%   rhs[leg]    = (J_B^T)^{-1} * tau_dyn[leg]      (3-vec, body frame)
%   lhs[leg]    = (J_B^T)^{-1} * tau_cmd[leg]      (3-vec, body frame)
%   F_body[leg] = LPF(rhs) - LPF(lhs)              (3-vec)
%
%   * tau_dyn  : 정역학 / RNEA 결과 (관절토크), 활성 3-DOF 만 사용
%   * tau_cmd  : 모터에 인가된 토크 명령 (본 프로젝트는 ctrl_input 으로 대체)
%   * J_B      : body frame 발 Jacobian (3x3, 활성 3-DOF)
%
% 직관 :
%   J_B^T * F_ext = tau_dyn - tau_cmd  (모터가 실제로 발휘한 토크와 동역학이
%                                       요구하는 토크의 차이가 외력)
%   따라서 F_ext = J_B^{-T} * (tau_dyn - tau_cmd) = rhs - lhs
%   고주파 잡음을 누르기 위해 양 항을 따로 LPF 후 차분.
%
% 내부 LPF :
%   Tustin 이산화 (util/filter.cc::lowpassfilter 와 동일 식) :
%     y_k = ( Ts * (x_k + x_{k-1}) - (Ts - 2 tau_c) y_{k-1} )
%                                 / (Ts + 2 tau_c)
%     where  tau_c = 1 / (2*pi*fc) = 1 / wc
%
% 첫 호출 (leg 별) 에는 LPF 상태를 정상상태값으로 seed 해서 transient 제거.

  properties (Access = private)
    dt_                              % 샘플 시간 (s)
    cutoff_hz_                       % 4 x 3 (leg/axis 별 LPF cutoff Hz)
    Jsing_thresh_                    % J^T 의 |det| 이 이 값 이하이면 0 출력

    % LPF 상태 (모두 3 x 4 행렬, 행=axis, 열=leg)
    rhs_         % 현재 step 입력
    rhs_old_     % 이전 step 입력
    lhs_
    lhs_old_
    rhs_lpf_     % 현재 step 출력
    rhs_lpf_old_ % 이전 step 출력
    lhs_lpf_
    lhs_lpf_old_

    initialized_     % leg 별 첫 호출 여부 (4x1 logical)
  end

  methods
    % -----------------------------------------------------------------
    % ctor
    %   dt           : 샘플 시간 (s)
    %   cutoff_hz    : 스칼라 또는 4x3.
    %                  스칼라면 모든 leg/axis 에 같은 cutoff 적용.
    %                  4x3 이면 행=leg, 열=axis 의 per-axis cutoff.
    %   Jsing_thresh : (선택, 기본 1e-6) J^T 특이성 보호
    % -----------------------------------------------------------------
    function obj = TaskspaceFOB(dt, cutoff_hz, Jsing_thresh)
      if nargin < 3 || isempty(Jsing_thresh), Jsing_thresh = 1e-6; end

      if isscalar(cutoff_hz)
        cutoff_hz = cutoff_hz * ones(4, 3);
      else
        assert(all(size(cutoff_hz) == [4 3]), ...
               'TaskspaceFOB: cutoff_hz must be scalar or 4x3.');
      end

      obj.dt_           = dt;
      obj.cutoff_hz_    = max(cutoff_hz, 1e-3);   % 0 또는 음수 방지 클램프
      obj.Jsing_thresh_ = Jsing_thresh;

      % LPF 상태 초기화
      Z = zeros(3, 4);
      obj.rhs_         = Z;  obj.rhs_old_     = Z;
      obj.lhs_         = Z;  obj.lhs_old_     = Z;
      obj.rhs_lpf_     = Z;  obj.rhs_lpf_old_ = Z;
      obj.lhs_lpf_     = Z;  obj.lhs_lpf_old_ = Z;
      obj.initialized_ = false(4, 1);
    end

    % -----------------------------------------------------------------
    % step : 한 시간 step 의 force 추정.
    %
    % 인자 :
    %   J_B_3x3x4    : leg 별 body-frame 발 Jacobian (3x3x4)
    %   tau_dyn_4x3  : leg 별 active joint 동역학 토크
    %                  행 = leg, 열 = HAA / HIP / DRIVER  [Nm]
    %   tau_cmd_4x3  : leg 별 active joint 명령 토크 (행/열 동일)
    %
    % 반환 :
    %   F_body_3x4   : leg 별 body-frame 외력 추정 (열이 한 leg)
    % -----------------------------------------------------------------
    function F_body_3x4 = step(obj, J_B_3x3x4, tau_dyn_4x3, tau_cmd_4x3)
      F_body_3x4 = zeros(3, 4);

      for leg = 1 : 4
        Jt  = J_B_3x3x4(:, :, leg)';        % 3x3, J^T

        % J^T 가 거의 특이행렬이면 안전하게 건너뜀
        if abs(det(Jt)) < obj.Jsing_thresh_
          continue;
        end

        tdyn = tau_dyn_4x3(leg, :)';         % 3x1
        tcmd = tau_cmd_4x3(leg, :)';         % 3x1

        % 새로운 입력 : J^T 의 역사상으로 task space 로 변환
        rhs_new = Jt \ tdyn;                 % == inv(Jt) * tdyn
        lhs_new = Jt \ tcmd;

        % 이전 step 값 갱신 (LPF 가 사용)
        obj.rhs_old_(:, leg)     = obj.rhs_(:, leg);
        obj.lhs_old_(:, leg)     = obj.lhs_(:, leg);
        obj.rhs_lpf_old_(:, leg) = obj.rhs_lpf_(:, leg);
        obj.lhs_lpf_old_(:, leg) = obj.lhs_lpf_(:, leg);

        obj.rhs_(:, leg) = rhs_new;
        obj.lhs_(:, leg) = lhs_new;

        % --- axis 별 Tustin LPF -----------------------------------------
        for ax = 1 : 3
          wc = 2 * pi * obj.cutoff_hz_(leg, ax);

          if ~obj.initialized_(leg)
            % 첫 호출 : LPF 상태를 정상상태값으로 seed (transient 제거)
            obj.rhs_lpf_(ax, leg)     = rhs_new(ax);
            obj.lhs_lpf_(ax, leg)     = lhs_new(ax);
            obj.rhs_lpf_old_(ax, leg) = rhs_new(ax);
            obj.lhs_lpf_old_(ax, leg) = lhs_new(ax);
          else
            obj.rhs_lpf_(ax, leg) = lpf_tustin(...
                obj.rhs_(ax, leg), obj.rhs_old_(ax, leg), ...
                obj.rhs_lpf_old_(ax, leg), wc, obj.dt_);
            obj.lhs_lpf_(ax, leg) = lpf_tustin(...
                obj.lhs_(ax, leg), obj.lhs_old_(ax, leg), ...
                obj.lhs_lpf_old_(ax, leg), wc, obj.dt_);
          end
        end
        obj.initialized_(leg) = true;

        % --- 최종 force = LPF(rhs) - LPF(lhs) ---------------------------
        F_body_3x4(:, leg) = obj.rhs_lpf_(:, leg) - obj.lhs_lpf_(:, leg);
      end
    end
  end
end

% =================================================================
% 로컬 유틸 : Tustin LPF (util/filter.cc::lowpassfilter 와 동일)
%
%   transfer fn :  Y(s)/X(s) = 1 / (s/wc + 1)
%   tau_c       =  1 / wc
%   y_k = ( Ts*(x_k + x_{k-1}) - (Ts - 2 tau_c) * y_{k-1} )
%                          / (Ts + 2 tau_c)
% =================================================================
function y = lpf_tustin(x, x_prev, y_prev, wc, dt)
  time_const = 1.0 / wc;     % wc = 2*pi*fc 이므로 시상수 = 1/wc
  y = (dt*(x + x_prev) - (dt - 2*time_const) * y_prev) / (dt + 2*time_const);
end
