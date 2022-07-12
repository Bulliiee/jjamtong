import RPi.GPIO as GPIO
from time import sleep

stop = 0
forward = 1
backword = 2
ch1=0
output = 1
input = 0

high = 1
low = 0

ena = 26

in1=19
in2=13

def setPinConfig(en, ina,inb):
    GPIO.setup(en,GPIO.OUT)
    GPIO.setup(ina, GPIO.OUT)
    GPIO.setup(inb, GPIO.OUT)
    pwm=GPIO.PWM(en,100)
    pwm.start(0)
    return pwm

def setMotorControl(pwm,ina,inb,speed, stat):
    pwm.ChangeDutyCycle(speed)
    
    if stat==forward:
        GPIO.output(ina,high)
        GPIO.output(inb,low)
        
    elif stat==stop:
        GPIO.output(ina,low)
        GPIO.output(inb,low)
        
def setMotor(ch,speed,stat):
    if ch==ch1:
        setMotorControl(pwmA,in1,in2,speed,stat)
        
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
pwmA=setPinConfig(ena,in1,in2)


setMotor(ch1,100,forward)
sleep(10)
setMotor(ch1,80,stop)

GPIO.cleanup()
    