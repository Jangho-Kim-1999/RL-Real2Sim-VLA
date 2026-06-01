function robot = load_urdf(urdfPath)
% SETUP_ROBOT  Quad URDF 를 fixed-base rigidBodyTree 로 로드.
%
% MATLAB importrobot 은 type="floating" joint 를 직접 지원하지 않는다.
% 이 함수는 URDF 텍스트에서 root_joint 의 floating 을 fixed 로 치환한 임시
% URDF 를 만든 뒤 importrobot 으로 로드한다 (base_link 가 world 에 고정).
%
% 본 프로젝트의 용도 (body-frame FK + body-frame Jacobian + 관절 토크 정역
% 계산) 에서 base 가 identity 로 고정되어도 정확한 결과를 얻을 수 있다.
% C++ 측의 compute_body_only 가 floating-base 부분을 강제 identity 로 덮어
% 쓴 뒤 FK 만 도는 것과 동일한 효과.
%
% 인자 :
%   urdfPath : Quad_v2_parallel.urdf 의 절대경로
% 반환 :
%   robot    : rigidBodyTree, DataFormat='column', Gravity=[0;0;-9.81]
%
% 동작 순서 :
%   1) URDF 텍스트 읽기
%   2) 'type="floating"' 을 'type="fixed"' 로 정규식 치환
%   3) 임시 파일에 저장 (원본 URDF 옆 우선, 실패 시 tempdir)
%   4) importrobot 으로 로드, DataFormat / Gravity 설정

  if ~exist(urdfPath, 'file')
    error('setup_robot:notFound', ...
          'URDF file not found at: %s', urdfPath);
  end

  % ---- 1) URDF 텍스트 읽기 + 2) 치환 ------------------------------------
  txt = fileread(urdfPath);
  txt = regexprep(txt, 'type="floating"', 'type="fixed"');

  % ---- 3) 임시 URDF 저장 ------------------------------------------------
  % 원본 URDF 와 같은 폴더에 저장하면 mesh 의 상대경로 ('./assets/meshes/...')
  % 가 그대로 유효하다. 쓰기 권한이 없으면 tempdir 로 fallback.
  [origDir, origName] = fileparts(urdfPath);
  tmpURDF = fullfile(origDir, [origName, '__fixedroot__.urdf']);
  fid = fopen(tmpURDF, 'w');
  if fid < 0
    tmpURDF = fullfile(tempdir, [origName, '__fixedroot__.urdf']);
    fid = fopen(tmpURDF, 'w');
    if fid < 0
      error('setup_robot:tempWrite', ...
            'Cannot write temp URDF.');
    end
  end
  fwrite(fid, txt);
  fclose(fid);

  % ---- 4) importrobot 로드 ----------------------------------------------
  % MeshPath 가 존재하면 visual mesh 도 같이 시도. 없거나 실패해도 FK/ID 는
  % 영향을 받지 않으므로 mesh 없이 재시도.
  meshPath = fullfile(fileparts(urdfPath), 'assets', 'meshes');
  try
    if exist(meshPath, 'dir')
      robot = importrobot(tmpURDF, 'MeshPath', meshPath);
    else
      robot = importrobot(tmpURDF);
    end
  catch ME
    warning('setup_robot:meshFail', ...
            'importrobot failed (%s) — retrying without mesh path.', ...
            ME.message);
    robot = importrobot(tmpURDF);
  end

  % column 형식 : q, qd, qdd 가 column vector 로 입출력 (이 프로젝트 가정).
  robot.DataFormat = 'column';
  % 중력 : world z-down. inverseDynamics 가 자동으로 G(q) 항에 반영.
  robot.Gravity    = [0; 0; -9.81];
end
