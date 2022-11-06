import RPi.GPIO as GPIO
import time
import os

in_pin = 23
out_pin = 24

GPIO.setmode(GPIO.BCM)

GPIO.setup(in_pin, GPIO.OUT)
GPIO.setup(out_pin, GPIO.OUT)

GPIO.output(in_pin, True)
#GPIO.output(out_pin, True)
input("filling --return to stop")
GPIO.output(in_pin, False)
#GPIO.output(out_pin, False)

print("taking img please wait...")
timestr = time.strftime("%Y%m%d-%H%M%S")
os.system (f"libcamera-still --tuning-file /usr/share/libcamera/ipa/raspberrypi/imx219_noir.json -o /home/pi/Desktop/log/current/{timestr}-raw.png")
print("done")


input("press return to empty")
GPIO.output(out_pin, True)
input("press return to stop")
GPIO.output(out_pin, False)
GPIO.cleanup()
