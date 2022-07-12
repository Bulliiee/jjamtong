# JjamTong


***Project description*** : This project is a garbage can that replaces recycling. When garbage is put in, whether it is cans, plastic, or regular garbage is separated through the camera. And the garbage recognized by the camera is transferred to the right container through the conveyor belt.

also check this gitlab URL: https://sw-git.chosun.ac.kr/20164338/jjamtong


___
***Project Features***

![캡처](/uploads/d97467e9325e31feb990491cddc09a80/캡처.PNG)


* YOLOv5 to learn the garbage to be collected.
* Compare the data learned through YOLOv5 with the data recognized by the camera and move the waste into the appropriate container.
* The garbage carried by the conveyor belt goes into the right trash can.
* If more than a certain amount of trash can is found through ultrasonic sensors, employees will be notified.
* In conjunction with the Android application, the user is notified.

***
***IDE, LANGUAGE***
* Application : android 10 or more, Android Studio, Java
* Raspberry Pi : Raspbian OS(64bit) , vsCode, Python3.7.0
* Web server : vsCode, Python3.7.0, html, Javascript

***
***YOLOv5 requirmets***
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


***Application requirments***
* Android sdk at least 27
* Google gms google service(firebase cloud message4.3.10)
* Google firebase-bom30.0.1
* Google firebase-analytics

***Raspberry Pi requirments***
* Cv2 => Same version as YOLOv5
* GPIO 

***Web requirments***
* Flask
* Cv2 => Same version as YOLOv5
* firebase admin
