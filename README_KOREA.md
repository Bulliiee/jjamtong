# 짬통


***프로젝트 설명*** : 이 프로젝트는 분리수거를 대신해주는 프로젝트이다. 쓰레기를 넣으면 캔이든지 플라스틱이든지 일반 쓰레기든지 학습받은 데이터와 비교하여 카메라를 통해 분리된다. 그리고 카메라로 인식된 쓰레기는 컨베이어 벨트를 통해 알맞은 쓰레기통으로 옮겨진다.

깃랩 참고: https://sw-git.chosun.ac.kr/20164338/jjamtong

___
***프로젝트 기능***

![캡처](/uploads/d97467e9325e31feb990491cddc09a80/캡처.PNG)


* YOLOv5로 수집한 데이터들을 학습 시킨다.
* YOLOv5를 통해 학습한 데이터와 카메라가 인식한 데이터를 비교하여 폐기물을 적절한 쓰레기통으로 옮길수 있다.
* 컨베이어 벨트를 통해 옮겨진 쓰레기들은 알맞은 통으로 들어간다.
* 만약 쓰레기통의 용량이 70퍼 이상 찾다면 수거하시는 분들에게 어플리케이션을 통하여 알림이 가도록 한다.
* 어플리케이션을 통하여 수거하시는 분들이 알림을 받을 수 있다.

***
***통합개발환경, 언어***

* 어플리케이션 : android 10 이상 , Android Studio, Java
* 라즈베리 파이 : Raspbian OS(64bit) , vsCode, Python3.7.0
* 웹 서버 : vsCode, Python3.7.0, html, Javascript

***
***YOLOv5 필수사항***
* matplotlib>=3.2.2
* numpy>=1.18.5
* opencv-python>=4.1.1
* Pillow>=7.1.2
* PyYAML>=5.3.1
* requests>=2.23.0
* scipy>=1.4.1  # Google Colab version
* torch>=1.7.0
* torchvision>=0.8.1
* tqdm>=4.41.0
* tensorflow>=2.4.1  # TFLite export
* tensorflowjs>=3.9.0  # TF.js export


***어플리케이션 필수사항***
* Android sdk 최소 27
* Google gms google service(firebase cloud message4.3.10)
* Google firebase-bom30.0.1
* Google firebase-analytics

***라즈베리 파이 필수사항***
* Cv2 => YOLOv5와 버전 같음.
* GPIO 

***웹 필수사항***
* Flask
* Cv2 => YOLOv5와 버전 같음.
* firebase admin
