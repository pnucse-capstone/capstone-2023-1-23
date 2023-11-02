### 1. 프로젝트 소개

프로젝트 명 : LiDAR 센서 데이터 기반 3D 공간 구축
목적 : 실제 공간에서 LiDAR 센서를 탑재한 카트가 수집한 포인트 클라우드를 활용하여 3D 가상 공간 모델을 생성하고 Unity로 렌더링한다.

### 2. 팀소개

홍주혁, ido_01@pusan.ac.kr, 데이터 수집 장치 제작 및 SLAM 시스템 구축

남예진, nmyejin@pusan.ac.kr, Rectangle to Cube 변환 및 벽과 장애물 구분 프로그램 개발

우현우, whw6578@pusan.ac.kr, Rectangle 추출 프로그램 개발

### 3. 시스템 구성도

프로젝트의 시스템 구성도는 다음과 같다.

![시스템 구성도 이미지](https://github.com/pnucse-capstone/capstone-2023-1-23/assets/102347501/cce5835a-aa81-42cc-a00b-b4a30f43d681)

#### 데이터 수집 장치
![카트 이미지](https://github.com/pnucse-capstone/capstone-2023-1-23/assets/102347501/57fa0e79-2b6d-497d-b5e1-2e2e3637af2d)
9축 기울기 센서와 키보드 조작을 통해 움직임을 추적하고 LiDAR 센서로 포인트 클라우드를 생성하여 SLAM을 진행한다.

Raspberry Pi에서는 PC의 rospy와 TCP 통신을 진행하고, 센서 값과 이동 신호를 주고 받는다. 이동 신호에 따라 모터를 움직이는 신호를 모터드라이버에 보낸다.

#### Rectangle 추출 프로그램
데이터 수집 장치에서 수집한 포인트 클라우드를 다음의 과정을 거쳐 Rectangles로 변환한다.

1. 데이터 전처리
2. 포인트 그룹화
3. Rectangle 변환
4. Rectangle 합병

#### 공간 장애물 구분 및 유니티 시각화
추출한 rectangle들을 읽고, convex hull과 여러 변환 과정을 거쳐 벽과 장애물 점을 분리한다.
장애물 점들은 거리로 분류한 후 벽과 같은 과정을 거쳐 벽과 장애물을 Cube로 생성한다.

### 4. 소개 및 시연 영상

[![유튜브 영상](http://img.youtube.com/vi/_DUv0jyv7Tc/0.jpg)](https://www.youtube.com/watch?v=_DUv0jyv7Tc)

### 5. 설치 및 사용법

본 프로젝트는 데이터 수집 장치로부터 수집된 pcd 확장자 파일을 Rectangle 추출 프로그램, Unity 시각화 프로그램을 거쳐 최종적인 렌더링을 수행한다.

#### Rectangle 추출 프로그램
해당 프로그램은 python3.11과 매트랩 프로그램, python의 매트랩 패키지와 매트랩 엔진 패키지가 필요하다.

매트랩 엔진 패키지 설치

```
pip install matlab matlabengine
```

데이터 수집 장치를 통해 생성된 데이터를 inputFiles 폴더에 pcd 파일로 넣는다.

프로그램 디렉토리에서 다음 명령어를 입력한다.
1. python main.py
2. inputFiles 폴더안의 파일 중 입력 파일 이름을 입력
3. 출력 방식을 1, 2, 3, 4중 선택하여 입력
4. 반복 횟수를 입력

출력 방식을 3번으로 선택 시 처리 과정별 fig 및 결과 텍스트 파일이 result 폴더에 저장된다. 
4번으로 선택 시 최종 결과 fig 및 결과 텍스트 파일이 result 폴더에 저장된다.

#### Unity 시각화 프로그램 
해당 프로그램은 KartLiDAR_23_build.zip을 다운받아 사용할 수 있다.

1. 다운로드 후 압축을 풀고 KartLiDAR_23.exe를 실행한다.

![시작 화면](https://github.com/pnucse-capstone/capstone-2023-1-23/assets/114997956/ede1e939-0f63-4073-80fb-f3ff1e50c437)

2. 3개의 입력란을 채운다.

   * file Name: 파일 경로/파일이름.txt (example1.txt, example2.txt, example3.txt를 입력하면 예시 파일을 사용할 수 있다.)

   * wall thickness: 벽 두께

   * scale constant: 스케일 조절
  
3. BUILD 버튼을 누른다.

![메인 화면1](https://github.com/pnucse-capstone/capstone-2023-1-23/assets/114997956/60871115-403c-4dbf-b373-f35d4a347151)
![메인 화면2](https://github.com/pnucse-capstone/capstone-2023-1-23/assets/114997956/ef0c831f-9f29-4102-a675-188754b81c87)

4. 오른쪽 아래의 OFF/ON 버튼을 누르면 장애물을 숨기고/나타낼 수 있다.

   키보드 방향키로 카메라의 위치와 각도를, 마우스 스크롤로 확대와 축소를 조절할 수 있다.
   
5. 이전 화면으로 돌아가려면 왼쪽 위의 ← 버튼을 누른다.
6. 프로그램을 종료하려면 esc 키를 누른다.
