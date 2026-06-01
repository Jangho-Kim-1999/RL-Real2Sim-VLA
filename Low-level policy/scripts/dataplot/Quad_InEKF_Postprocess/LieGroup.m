classdef LieGroup
% LIEGROUP  InEKF에 필요한 Lie 군 연산 모음 (정적 함수만 보유).
%
% Biarticular_Quad_Ctrl/util/LieGroup.cc 의 직접 포팅.
%
% 제공 함수 :
%   - skew(v)            : 3차원 벡터 -> 3x3 비대칭 행렬 (× 연산자)
%   - ExpSO3(w)          : SO(3) 의 지수 사상 (Rodrigues 공식)
%   - ExpSEK3(v)         : SE_K(3) 의 지수 사상.
%                          v 의 크기 = 3 + 3*K (회전 1 블록 + 평행이동 K 블록)
%                          반환 X 의 크기 = (3+K) x (3+K)
%   - AdjointSEK3(X)     : SE_K(3) 원소의 adjoint 행렬
%
% 모든 함수가 static 이므로 클래스 자체는 namespace 역할.

  properties (Constant, Access = private)
    % 회전각 theta -> 0 근처에서 분기 처리하는 임계값.
    kTolerance = 1e-10;
  end

  methods (Static)

    % ---------------------------------------------------------------
    % skew :  v = [v1; v2; v3]  ->  [  0, -v3,  v2 ;
    %                                  v3,  0, -v1 ;
    %                                 -v2,  v1,  0  ]
    % 외적 a × b = skew(a) * b 로 표현하기 위한 기본 도구.
    % ---------------------------------------------------------------
    function M = skew(v)
      M = [    0,   -v(3),  v(2);
            v(3),     0,   -v(1);
           -v(2),   v(1),    0];
    end

    % ---------------------------------------------------------------
    % ExpSO3 : 축-각 벡터 w (3x1) -> 회전행렬 R (3x3)
    %
    % R = I + (sin(theta)/theta) * [w]× + ((1-cos(theta))/theta^2) * [w]×^2
    %
    % theta = ||w|| 가 0 근처면 R = I 로 처리 (분모 0 방지).
    % ---------------------------------------------------------------
    function R = ExpSO3(w)
      tol   = 1e-10;
      theta = norm(w);

      if theta < tol
        R = eye(3);
        return;
      end

      A = LieGroup.skew(w);
      R = eye(3) ...
        + (sin(theta)/theta)         * A ...
        + ((1 - cos(theta))/(theta^2)) * (A * A);
    end

    % ---------------------------------------------------------------
    % ExpSEK3 : SE_K(3) 의 지수 사상.
    %
    % 입력 v 의 형식 (총 3 + 3*K 원소) :
    %   v = [ phi (3) ; rho_1 (3) ; rho_2 (3) ; ... ; rho_K (3) ]
    %
    % 출력 X 는 (3+K) x (3+K) 의 SE_K(3) 행렬 :
    %   X = [ R    Jl*rho_1   Jl*rho_2   ...   Jl*rho_K ;
    %         0      1            0       ...     0     ;
    %         0      0            1       ...     0     ;
    %         ...                                        ;
    %         0      0            0       ...     1     ]
    %
    %   R  = Exp_SO3(phi)
    %   Jl = SO(3) 의 left Jacobian at phi
    %      = I + ((1-cos)/theta^2) [phi]× + ((theta-sin)/theta^3) [phi]×^2
    %
    % 회전 0 근처에서는 R = I, Jl = I 로 분기.
    %
    % InEKF 안에서의 사용 : correct() 가 update vector delta 를 X 에
    % left-multiplication 으로 합칠 때 dX = ExpSEK3(delta_X) 로 변환.
    % ---------------------------------------------------------------
    function X = ExpSEK3(v)
      tol = 1e-10;
      n   = numel(v);
      assert(mod(n, 3) == 0 && n >= 3, 'ExpSEK3: v size must be 3*(K+1)');
      K   = (n - 3) / 3;     % 평행이동 블록 개수

      X = eye(3 + K);

      w     = v(1:3);
      theta = norm(w);

      if theta < tol
        % theta -> 0 극한 : R = I, Jl = I
        Rmat = eye(3);
        Jl   = eye(3);
      else
        A      = LieGroup.skew(w);
        theta2 = theta^2;
        s      = sin(theta);
        c      = cos(theta);
        oneMc  = (1 - c) / theta2;
        A2     = A * A;

        Rmat   = eye(3) + (s/theta) * A + oneMc * A2;
        Jl     = eye(3) + oneMc      * A + ((theta - s)/(theta2 * theta)) * A2;
      end

      % 회전 블록
      X(1:3, 1:3) = Rmat;

      % K 개의 평행이동 블록 : 각각 Jl * rho_i
      for i = 1 : K
        rho = v(3 + 3*(i-1) + 1 : 3 + 3*i);
        X(1:3, 3 + i) = Jl * rho;
      end
    end

    % ---------------------------------------------------------------
    % AdjointSEK3 : SE_K(3) 원소 X 의 adjoint 행렬.
    %
    % X 가 (3+K) x (3+K) 이면 Adj 는 (3 + 3K) x (3 + 3K).
    %
    % 블록 구조 :
    %   Adj = [    R              0     0    ...    0    ;
    %           [d_1]× R          R     0    ...    0    ;
    %           [d_2]× R          0     R    ...    0    ;
    %             ...                                     ;
    %           [d_K]× R          0     0    ...    R   ]
    %
    % InEKF propagate 에서 process noise 를 right-invariant frame 으로
    % 옮기는 데 사용 : Q_hat = Phi * Adj * Q * Adj' * Phi' * dt
    % ---------------------------------------------------------------
    function Adj = AdjointSEK3(X)
      K   = size(X, 2) - 3;
      Adj = zeros(3 + 3*K, 3 + 3*K);

      Rmat = X(1:3, 1:3);

      % 좌상단 회전 블록
      Adj(1:3, 1:3) = Rmat;

      % K 개의 평행이동 블록 + 회전과의 결합 [d_i]× R
      for i = 1 : K
        di     = X(1:3, 3 + i);
        rowOff = 3 + 3*(i-1);
        Adj(rowOff+1 : rowOff+3, rowOff+1 : rowOff+3) = Rmat;                  % 대각
        Adj(rowOff+1 : rowOff+3, 1:3)                 = LieGroup.skew(di) * Rmat;
      end
    end

  end
end
