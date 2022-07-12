import os,sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from yolov5_custom import detect_copy

# print(sys.path)
detect_copy.yolo_main()