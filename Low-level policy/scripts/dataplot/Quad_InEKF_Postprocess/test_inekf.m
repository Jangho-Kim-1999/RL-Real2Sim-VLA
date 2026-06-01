% TEST_INEKF  본 estimation 파이프라인 실행 전, EKF 클래스만 단독으로 동작 점검. 
% LieGroup 연산, 정지 IMU 입력 propagation, contact augment / remove cycle 까지 3 단계 검증.
%
% 같은 폴더에 InvariantEKF.m, LieGroup.m 이 있어야 한다.
% MATLAB 콘솔에서 : test_inekf 실행.
% 모든 assert 가 통과하면 "All smoke tests passed."가 출력된다.

function test_inekf()

  % =========================================================================
  % STAGE 1 : LieGroup 의 기본 성질을 잘 만족하는지 확인
  % =========================================================================
  fprintf('Stage 1 : LieGroup math\n');
  v = [0.1; 0.2; 0.3];

  % Exp_SO3 : 결과가 회전행렬 (det=1, 직교) 이어야 함
  R = LieGroup.ExpSO3(v);
  assert(abs(det(R) - 1) < 1e-10,            'ExpSO3 determinant');
  assert(norm(R*R' - eye(3), 'fro') < 1e-10, 'ExpSO3 orthogonality');

  % ExpSEK3 : v 길이 9 = 1 phi + 2 rho 블록  ->  X 는 5x5
  X = LieGroup.ExpSEK3([v; 1; 2; 3; 4; 5; 6]);
  assert(all(size(X) == [5 5]),                       'ExpSEK3 size');
  assert(abs(X(4,4) - 1) < 1e-10 && ...
         abs(X(5,5) - 1) < 1e-10,                     'ExpSEK3 trailing');

  % Adjoint : K=2 -> Adj 는 (3+3*2) x (3+3*2) = 9x9
  Adj = LieGroup.AdjointSEK3(X);
  assert(all(size(Adj) == [9 9]),                     'AdjointSEK3 size');

  % =========================================================================
  % STAGE 2 : 정지 IMU 입력에서 X 가 drift 안 하는지 확인.
  % gyro=0, accel=(0,0,9.81). 1 s 동안 propagate 했을 때 R=I, v=0, p 유지해야 함.
  % =========================================================================
  fprintf('Stage 2 : InvariantEKF with stationary input\n');
  ekf = InvariantEKF(struct('p', [0;0;0.3]));
  ekf.setNoise(0.01, 0.1, 1e-5, 1e-4, 0.01);

  gyro  = [0;0;0];
  accel = [0;0;9.81];        % R*a + g_W = (0,0,9.81) + (0,0,-9.81) = 0
  dt    = 0.001;

  for k = 1 : 1000
    ekf.propagate(gyro, accel, dt);
  end

  R = ekf.getR(); v = ekf.getV(); p = ekf.getP();
  assert(norm(R - eye(3), 'fro') < 1e-6, 'static R drift');
  assert(norm(v) < 1e-4,                  'static v drift');
  assert(norm(p - [0;0;0.3]) < 1e-4,      'static p drift');

  % =========================================================================
  % STAGE 3 : kinematic correct + augment + remove cycle가 제대로 작동하는지 확인
  %
  % 1) 4 leg 모두 stance -> 모두 augment, contact_slot 이 (6,7,8,9) 가 되어야 한다.
  % 2) leg 2 만 swing 으로 빠짐 -> leg 2 의 slot 만 -1 로, 나머지 유지.
  % =========================================================================
  fprintf('Stage 3 : kinematic correct + augment + remove\n');

  % 4개 leg가 모두 stance인 경우 augment, contact_slot 이 (6,7,8,9)가 되는지 확인.
  contact = [true; true; true; true];
  foot_pos_body = [0.2  0.2 -0.2 -0.2;       % leg 별 발 위치 임의로 설정 (body frame)
                   0.1 -0.1  0.1 -0.1;
                  -0.3 -0.3 -0.3 -0.3];

  ekf.correctKinematics(contact, foot_pos_body);
  s = ekf.getSlots();
  assert(all(s == [6;7;8;9]), 'augment slots');

  % leg 2 만 swing 으로 빠지면 -> leg 2 만 -1로 바뀌고, 나머지는 살아남는지 확인.
  contact(2) = false;
  ekf.correctKinematics(contact, foot_pos_body);
  s = ekf.getSlots();
  assert(s(2) == -1,            'leg 2 removed');
  assert(all(s([1 3 4]) > 0),   'others retained');

  fprintf('All smoke tests passed.\n');
end
