# import RPi.GPIO as GPIO
# import time

# def sonic():
# 	GPIO.setmode(GPIO.BCM)
# 	GPIO.setwarnings(False)
# 	c=0
# 	TRIG1=23
# 	ECHO1=24
# 	TRIG2=17
# 	ECHO2=27
# 	TRIG3=22
# 	ECHO3=10
# 	TRIG4=9
# 	ECHO4=11
# 	TRIG5=25
# 	ECHO5=8
# 	TRIG6=7
# 	ECHO6=12
# 	GPIO.setup(TRIG1,GPIO.OUT)
# 	GPIO.setup(ECHO1,GPIO.IN)
# 	GPIO.setup(TRIG2,GPIO.OUT)
# 	GPIO.setup(ECHO2,GPIO.IN)
# 	GPIO.setup(TRIG3,GPIO.OUT)
# 	GPIO.setup(ECHO3,GPIO.IN)
# 	GPIO.setup(TRIG4,GPIO.OUT)
# 	GPIO.setup(ECHO4,GPIO.IN)
# 	GPIO.setup(TRIG5,GPIO.OUT)
# 	GPIO.setup(ECHO5,GPIO.IN)
# 	GPIO.setup(TRIG6,GPIO.OUT)
# 	GPIO.setup(ECHO6,GPIO.IN)
 
# 	GPIO.output(TRIG1,False)
# 	GPIO.output(TRIG2,False)
# 	GPIO.output(TRIG3,False)
# 	GPIO.output(TRIG4,False)
# 	GPIO.output(TRIG5,False)
# 	GPIO.output(TRIG6,False)
# 	print("wait")
# 	time.sleep(0.5)

# 	try:
# 		while c<2:
# 			GPIO.output(TRIG1,True)
# 			time.sleep(1)
# 			GPIO.output(TRIG1,False)
		
# 			while GPIO.input(ECHO1)==0:
# 				start1=time.time()
# 			while GPIO.input(ECHO1)==1:
# 				stop1=time.time()	
		
# 			check_time1=stop1-start1
# 			can1=check_time1*34300/2
# 			print("can1: %.1f cm"%can1)
		
# 			GPIO.output(TRIG2,True)
# 			time.sleep(1)
# 			GPIO.output(TRIG2,False)
		
# 			while GPIO.input(ECHO2)==0:
# 				start2=time.time()
# 			while GPIO.input(ECHO2)==1:
# 				stop2=time.time()	
		
# 			check_time2=stop2-start2
# 			can2=check_time2*34300/2
# 			print("can2: %.1f cm"%can2)

# 			GPIO.output(TRIG3,True)
# 			time.sleep(1)
# 			GPIO.output(TRIG3,False)
		
# 			while GPIO.input(ECHO3)==0:
# 				start3=time.time()
# 			while GPIO.input(ECHO3)==1:
# 				stop3=time.time()	
		
# 			check_time3=stop3-start3
# 			pet1=check_time3*34300/2
# 			print("pet1: %.1f cm"%pet1)
   
# 			GPIO.output(TRIG4,True)
# 			time.sleep(1)
# 			GPIO.output(TRIG4,False)
		
# 			while GPIO.input(ECHO4)==0:
# 				start4=time.time()
# 			while GPIO.input(ECHO4)==1:
# 				stop4=time.time()	
		
# 			check_time4=stop4-start4
# 			pet2=check_time4*34300/2
# 			print("pet2: %.1f cm"%pet2)
	
# 			GPIO.output(TRIG5,True)
# 			time.sleep(1)
# 			GPIO.output(TRIG5,False)
		
# 			while GPIO.input(ECHO5)==0:
# 				start5=time.time()
# 			while GPIO.input(ECHO5)==1:
# 				stop5=time.time()	
		
# 			check_time5=stop5-start5
# 			trs1=check_time5*34300/2
# 			print("trs1: %.1f cm"%trs1)
   
# 			GPIO.output(TRIG6,True)
# 			time.sleep(1)
# 			GPIO.output(TRIG6,False)
		
# 			while GPIO.input(ECHO6)==0:
# 				start6=time.time()
# 			while GPIO.input(ECHO6)==1:
# 				stop6=time.time()	
		
# 			check_time6=stop6-start6
# 			trs2=check_time6*34300/2
# 			print("trs2: %.1f cm"%trs2)

# 			c=c+1
# 	except KeyboardInterrupt:
# 		print("stop")
# 		GPIO.cleanup()
import pigpio
from time import sleep
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
c=0
TRIG1=23
ECHO1=24
TRIG2=27
ECHO2=17
TRIG3=22
ECHO3=10
TRIG4=20
ECHO4=21
TRIG5=6
ECHO5=5
TRIG6=16
ECHO6=26
GPIO.setup(TRIG1,GPIO.OUT)
GPIO.setup(ECHO1,GPIO.IN)
GPIO.setup(TRIG2,GPIO.OUT)
GPIO.setup(ECHO2,GPIO.IN)
GPIO.setup(TRIG3,GPIO.OUT)
GPIO.setup(ECHO3,GPIO.IN)
GPIO.setup(TRIG4,GPIO.OUT)
GPIO.setup(ECHO4,GPIO.IN)
GPIO.setup(TRIG5,GPIO.OUT)
GPIO.setup(ECHO5,GPIO.IN)
GPIO.setup(TRIG6,GPIO.OUT)
GPIO.setup(ECHO6,GPIO.IN)

GPIO.output(TRIG1,False)
GPIO.output(TRIG2,False)
GPIO.output(TRIG3,False)
GPIO.output(TRIG4,False)
GPIO.output(TRIG5,False)
GPIO.output(TRIG6,False)

print("wait")
#time.sleep(0.5)


pi=pigpio.pi()

# pi.set_servo_pulsewidth(18,0)
sleep(1)
pi.set_servo_pulsewidth(18,1100)
sleep(1)
pi.set_servo_pulsewidth(18,1800)
sleep(1)
try:
	while True:
		GPIO.output(TRIG1,True)
		time.sleep(0.5)
		GPIO.output(TRIG1,False)
	
		while GPIO.input(ECHO1)==0:
			start1=time.time()
		while GPIO.input(ECHO1)==1:
			stop1=time.time()
		
		check_time1=stop1-start1
		can1=check_time1*34300/2
		print("can1: %.1f cm"%can1)
		
		GPIO.output(TRIG2,True)
		time.sleep(0.5)
		GPIO.output(TRIG2,False)
	
		while GPIO.input(ECHO2)==0:
			start2=time.time()
		while GPIO.input(ECHO2)==1:
			stop2=time.time()
		
		check_time2=stop2-start2
		can2=check_time2*34300/2
		print("can2: %.1f cm"%can2)
		
		GPIO.output(TRIG3,True)
		time.sleep(0.5)
		GPIO.output(TRIG3,False)
	
		while GPIO.input(ECHO3)==0:
			start3=time.time()
		while GPIO.input(ECHO3)==1:
			stop3=time.time()
		
		check_time3=stop3-start3
		pet1=check_time3*34300/2
		print("pet1: %.1f cm"%pet1)
		
		GPIO.output(TRIG4,True)
		time.sleep(0.5)
		GPIO.output(TRIG4,False)
	
		while GPIO.input(ECHO4)==0:
			start4=time.time()
		while GPIO.input(ECHO4)==1:
			stop4=time.time()
		
		check_time4=stop4-start4
		pet2=check_time4*34300/2
		print("pet2: %.1f cm"%pet2)
		
		GPIO.output(TRIG5,True)
		time.sleep(0.5)
		GPIO.output(TRIG5,False)
		while GPIO.input(ECHO5)==0:
			start5=time.time()
		while GPIO.input(ECHO5)==1:
			stop5=time.time()
			
		check_time5=stop5-start5
		trs1=check_time5*34300/2
		print("trs1: %.1f cm"%trs1)
		
		GPIO.output(TRIG6,True)
		time.sleep(0.5)
		GPIO.output(TRIG6,False)
	
		while GPIO.input(ECHO6)==0:
			start6=time.time()
		while GPIO.input(ECHO6)==1:
			stop6=time.time()
		
		check_time6=stop6-start6
		trs2=check_time6*34300/2
		print("trs2: %.1f cm"%trs2)
		
		
except KeyboardInterrupt:
	print("stop")
	GPIO.cleanup()