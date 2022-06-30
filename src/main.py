from yolo import Yolo
import cv2
import time
import argparse
from os import path

# usage: python main.py --image your_image.png

if __name__=="__main__":
    
    YOLO_CFG_PATH = "../yolo-cfg/yolov4-tiny-custom.cfg"
    YOLO_WEIGHTS_PATH = "../backup/yolov4-tiny-custom_final.weights"
    YOLO_CLASS_NAMES_PATH = "../class.names"
    
    parser = argparse.ArgumentParser(description='Object Detection using YOLOV4')
    parser.add_argument('--image', help='Path to image file.')
    
    args = parser.parse_args()
    
    if path.exists(args.image):
        img = cv2.imread(args.image)
    else:
        raise SystemExit("Error loading the image!")
    
    labelsFile = open(YOLO_CLASS_NAMES_PATH, "r")
    labelsList = [label.strip() for label in labelsFile.readlines()]

    model = Yolo(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH, labelsList, (416, 416), 0.3)
    
    start = time.time()
    result = model.inference(img)
    end = time.time()
    
    print("Inference Time: ", end-start)

    for det in result:
        cv2.rectangle(img, det["roi"], (0,255,0))
    cv2.imwrite("det.png", img)