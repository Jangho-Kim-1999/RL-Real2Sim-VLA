function map = build_joint_map(robot)
% BUILD_JOINT_MAP  Quad rigidBodyTree 의 leg 별 joint 인덱스 매핑 생성.
%
% setup_robot 으로 root_joint 를 fixed 로 바꾸고 나면 로봇은 20 개의
% revolute joint 만 갖게 된다 (leg 당 5 개 × 4 leg). 이때 q-vector 의 차원도
% 20 이지만, MATLAB importrobot 이 URDF tree 를 깊이우선 순회하면서 어느
% 순서로 joint 를 배치할지는 미리 알 수 없다.
%
% 이 함수는 robot.Bodies 를 직접 순회해서, "joint 이름" -> "q-vector 인덱스"
% 의 매핑을 만들고 leg 4 개 × {active 3 + passive 2} 로 정리한다.
%
% 반환 struct 의 필드 :
%   map.q_active_idx  (4x3) : leg 별 active joint q-index
%                              (열 1=HAA, 2=HIP, 3=DRIVER)
%   map.q_passive_idx (4x2) : leg 별 passive joint q-index
%                              (열 1=KNEE, 2=COUPLING)
%   map.foot_frame    {1x4} : leg 별 발 frame 이름
%   map.imu_frame     str   : IMU frame 이름
%   map.leg_names     {1x4} : 'FL','FR','RL','RR' (참고용)
%   map.q_dim         int   : 로봇의 nv (= homeConfiguration 의 길이)
%   map.joint_names   string array : q-index 순서대로 joint 이름
%
% URDF 명명 규칙 (this robot 한정) :
%   prefix in {FL, FR, RL, RR}
%   active  : <prefix>HAA, <prefix>HIP, <prefix>_driver_input
%   passive : <prefix>KNEE, <prefix>_input_coupling

  legPrefix     = {'FL', 'FR', 'RL', 'RR'};
  activeSuffix  = {'HAA',  'HIP',   '_driver_input'};
  passiveSuffix = {'KNEE', '_input_coupling'};

  % ---- 로봇의 모든 non-fixed joint 를 q-index 순서로 수집 ---------------
  % homeConfiguration 의 길이가 곧 nv. 이 길이가 우리가 찾는 q_dim.
  q_dim = numel(homeConfiguration(robot));

  % robot.Bodies 의 순회 순서가 곧 q-index 의 순서 (DataFormat='column' 일 때).
  jointName = strings(q_dim, 1);
  idx = 1;
  for b = 1 : numel(robot.Bodies)
    j = robot.Bodies{b}.Joint;
    if ~strcmp(j.Type, 'fixed')
      jointName(idx) = string(j.Name);
      idx = idx + 1;
    end
  end

  % 누락 검사 : 20 이 아닐 경우 URDF 가 예상과 다르다는 의미.
  if idx-1 ~= q_dim
    error('build_joint_map:dimMismatch', ...
          'Counted %d non-fixed joints but q_dim = %d.', idx-1, q_dim);
  end

  % ---- leg 별 active / passive joint 인덱스 매핑 ------------------------
  q_active_idx  = zeros(4, 3);
  q_passive_idx = zeros(4, 2);

  for L = 1 : 4
    pre = legPrefix{L};

    % active 3 개 (HAA, HIP, DRIVER)
    for k = 1 : 3
      target = [pre, activeSuffix{k}];
      hit = find(jointName == string(target), 1);
      if isempty(hit)
        error('build_joint_map:notFound', ...
              'Active joint "%s" not found in robot.', target);
      end
      q_active_idx(L, k) = hit;
    end

    % passive 2 개 (KNEE, COUPLING)
    for k = 1 : 2
      target = [pre, passiveSuffix{k}];
      hit = find(jointName == string(target), 1);
      if isempty(hit)
        error('build_joint_map:notFound', ...
              'Passive joint "%s" not found in robot.', target);
      end
      q_passive_idx(L, k) = hit;
    end
  end

  % ---- 출력 struct 조립 -------------------------------------------------
  map.q_active_idx  = q_active_idx;
  map.q_passive_idx = q_passive_idx;
  map.foot_frame    = {'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'};
  map.imu_frame     = 'imu_link';
  map.leg_names     = legPrefix;
  map.q_dim         = q_dim;
  map.joint_names   = jointName;
end
