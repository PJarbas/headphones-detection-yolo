Training YOLOv3 Object Detector - Snowman

## Headphones object detection using Yolo

This project aims to train a object detecion using custom number of classes (Headphones)


### Get started

Ensure that your **python** version is >= 3.9

Download the data running the following command in the `src` directory:

```bash
   $ python data_downloader.py 
```


Create the train.txt and test.txt files with random split

```bash
   $ python split_train_test.py ../data/images/ 
```


### Install YoloV4 Darknet:


* https://medium.com/geekculture/yolov4-darknet-installation-and-usage-on-your-system-windows-linux-8dec2cea6e81


Get the yolov4-tiny.cfg and change the number of classes and the filters for the [conv] layers before the [yolo] layers 

```bash
   $ wget https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg
```

### Train

```bash
   $ darknet detector train yolo-cfg/darknet.data yolo-cfg/yolov4-tiny.cfg  
```

6. Give the correct path to the modelConfiguration and modelWeights files in object_detection_yolo.py and test any image or video for snowman detection, e.g.

`python3 object_detection_yolo.py --image=snowmanImage.jpg`


