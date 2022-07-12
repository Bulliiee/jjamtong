# YOLOv5 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
# from gpio import sonic
import pigpio
import argparse
import os
import sys
from pathlib import Path
import RPi.GPIO as GPIO
import time
import torch
import torch.backends.cudnn as cudnn
import socket
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit
import socketio

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

GPIO.setmode(GPIO.BCM)  # Use Board numerotation mode
GPIO.setwarnings(False)  # Disable warnings


# def angle_to_percent(angle):  # Use servo angle
#     if angle > 180 or angle < 0:
#         return False

#     start = 5
#     end = 12
#     ratio = (end - start) / 180  # Calcul ratio from angle to percent

#     angle_as_percent = angle * ratio

#     return start + angle_as_percent


# Use pin 12 for PWM signal
sonictime = 1800
can1 = 18
pet1 = 13
servo=pigpio.pi()
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
GPIO.setup(TRIG1, GPIO.OUT)
GPIO.setup(ECHO1, GPIO.IN)
GPIO.setup(TRIG2, GPIO.OUT)
GPIO.setup(ECHO2, GPIO.IN)
GPIO.setup(TRIG3, GPIO.OUT)
GPIO.setup(ECHO3, GPIO.IN)
GPIO.setup(TRIG4, GPIO.OUT)
GPIO.setup(ECHO4, GPIO.IN)
GPIO.setup(TRIG5, GPIO.OUT)
GPIO.setup(ECHO5, GPIO.IN)
GPIO.setup(TRIG6, GPIO.OUT)
GPIO.setup(ECHO6, GPIO.IN)

GPIO.output(TRIG1, False)
GPIO.output(TRIG2, False)
GPIO.output(TRIG3, False)
GPIO.output(TRIG4, False)
GPIO.output(TRIG5, False)
GPIO.output(TRIG6, False)
# frequence = 50
# GPIO.setup(can1, GPIO.OUT)
# GPIO.setup(pet1, GPIO.OUT)
# can = GPIO.PWM(can1, frequence)
# pet = GPIO.PWM(pet1, frequence)
# can.start(0)
# pet.start(0)
#global scan1, scan2, spet1, spet2, strs1, strs2
HOST = '192.168.1.10'
PORT = 8080



def webconnect():
    sonicsend = socketio.Client()
    sonicsend.connect('http://203.237.100.96:80', wait_timeout=10)
    return sonicsend


def websend(sonicsend, a, b, c, d, e, f):
    # emit('message',c)
    sonicsend.on('connect')
    sonicsend.emit('capacity', str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + ' ' + str(e) + ' ' + str(f))
    # sonicsend.emit('test',str(a))
    print("send")
    # s.disconnect()
    # data = s.recv(1024)
    # print ('result: ' + data.decode('utf-8'))


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=5,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    sonictime=1800
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            
            if sonictime >= 1800:
                print("\n\n\nchecksonic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")             
                can1cm, can2cm, pet1cm, pet2cm, trs1cm, trs2cm = soniccheck()
                sonictime = 0
                con = webconnect()
                websend(con, can1cm, can2cm, pet1cm, pet2cm, trs1cm, trs2cm)
                time.sleep(0.5)
               # print("\n\n\n\n\n" + can1cm+" "+can2cm + "\n\n\n\n\n")
                con.disconnect()
                time.sleep(0.5)
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class``

                # c is detected class number
                # 0: can, 1: pet
                
                LOGGER.info(int(c))
                # time.sleep(5)
                if c == 0:
                    print("close")
                    servo.set_servo_pulsewidth(can1,1400)
                    time.sleep(3)
                    servo.set_servo_pulsewidth(can1,780)
                    time.sleep(0.5)
                    servo.set_servo_pulsewidth(can1,0)
                if c == 1:
                    servo.set_servo_pulsewidth(pet1,1900)
                    print("this")
                    time.sleep(5)
                    servo.set_servo_pulsewidth(pet1,1100)
                    time.sleep(0.5)
                    servo.set_servo_pulsewidth(pet1,0)

            else:
                # -1: none
                trash = -1
            sonictime = sonictime + 10


# def open(pin):
#     if(pin==18):
#         return can.ChangeDutyCycle(angle_to_percent(180))
#         return can.ChangeDutyCycle(12.5)
#     else:
#         #return pet.ChangeDutyCycle(angle_to_percent(30))
#         return pet.ChangeDutyCycle(12.5)
# def close(pin):
#     if(pin==18):
#         return can.ChangeDutyCycle(angle_to_percent(10))
#         return can.ChangeDutyCycle(7.5)
#     else:
#         return pet.ChangeDutyCycle(angle_to_percent(70))
#         return pet.ChangeDutyCycle(7.5)


def soniccheck():
    GPIO.output(TRIG1, True)
    time.sleep(0.5)
    GPIO.output(TRIG1, False)
    while GPIO.input(ECHO1) == 0:
        sstart1 = time.time()
    while GPIO.input(ECHO1) == 1:
        sstop1 = time.time()
    scheck_time1 = sstop1 - sstart1
    scan1 = scheck_time1 * 34300 / 2
    scan1 = round(scan1)

    GPIO.output(TRIG2, True)
    time.sleep(0.5)
    GPIO.output(TRIG2, False)
    while GPIO.input(ECHO2) == 0:
        sstart2 = time.time()
    while GPIO.input(ECHO2) == 1:
        sstop2 = time.time()
    scheck_time2 = sstop2 - sstart2
    scan2 = scheck_time2 * 34300 / 2
    scan2 = round(scan2)

    GPIO.output(TRIG3, True)
    time.sleep(0.5)
    GPIO.output(TRIG3, False)
    while GPIO.input(ECHO3) == 0:
        sstart3 = time.time()
    while GPIO.input(ECHO3) == 1:
        sstop3 = time.time()
    scheck_time3 = sstop3 - sstart3
    spet1 = scheck_time3 * 34300 / 2
    spet1 = round(spet1)

    GPIO.output(TRIG4, True)
    time.sleep(0.5)
    GPIO.output(TRIG4, False)
    while GPIO.input(ECHO4) == 0:
        sstart4 = time.time()
    while GPIO.input(ECHO4) == 1:
        sstop4 = time.time()
    scheck_time4 = sstop4 - sstart4
    spet2 = scheck_time4 * 34300 / 2
    spet2 = round(spet2)

    GPIO.output(TRIG5, True)
    time.sleep(0.5)
    GPIO.output(TRIG5, False)
    while GPIO.input(ECHO5) == 0:
        sstart5 = time.time()
    while GPIO.input(ECHO5) == 1:
        sstop5 = time.time()
    scheck_time5 = sstop5 - sstart5
    strs1 = scheck_time5 * 34300 / 2
    strs1 = round(strs1)

    GPIO.output(TRIG6, True)
    time.sleep(0.5)
    GPIO.output(TRIG6, False)
    while GPIO.input(ECHO6) == 0:
        sstart6 = time.time()
    while GPIO.input(ECHO6) == 1:
        sstop6 = time.time()
    scheck_time6 = sstop6 - sstart6
    strs2 = scheck_time6 * 34300 / 2
    strs2 = round(strs2)
    print(str(scan1)+" "+ str(scan2)+" "+ str(spet1)+" "+ str(spet2)+" "+ str(strs1)+" "+ str(strs2))
    return scan1, scan2, spet1, spet2, strs1, strs2
    

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/exp4/weights/best.pt', help='model path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp2/weights/best.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default="http://203.237.100.96:8090/?action=stream", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default="http://203.237.100.96:8083//video", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 480], help='inference size h,w')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[384, 480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    # parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=5, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

# def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)

def yolo_main():
    LOGGER.info("ROOT>>")
    LOGGER.info(ROOT)

    LOGGER.info('get opt')
    opt = parse_opt()

    LOGGER.info('check requirement')
    check_requirements(exclude=('tensorboard', 'thop'))

    LOGGER.info('start')
    run(**vars(opt))

if __name__ == "__main__":
    yolo_main()