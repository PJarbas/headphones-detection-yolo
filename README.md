

## Headphones object detection using YoloV4

Headphones Detection 1     |  Headphones Detection 2   |  Headphones Detection 3 
:-------------------------:|:-------------------------:|:-------------------------:
![](icon/det.png)          | ![](icon/det1.png)        |  ![](icon/det2.png)


This project aims to train a object detecion deep learning model, using a custom number of classes (Headphones)


### Get started

Ensure that your **python** version is >= 3.9 and you have installed the opencv with CUDA

Download the data running the following command in the `src` directory:

```bash
   $ python data_downloader.py 
```


Create the `train.txt` and `test.txt` files with random split, these files must be in the same directory that the `darknet.data` file

```bash
   $ cd src
   $ python split_train_test.py ../data/images/ 
```


### Install YoloV4 Darknet with CUDA:

* https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg


### Download the yolov4-tiny-custom

Get the yolov4-tiny-custom.cfg and change the number of classes and the filters for the [conv] layers before the [yolo] layers 

* check the documentation to see all the parameters to be changed: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

```bash
   $ wget https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-custom.cfg
```

### Train using pre-trained weights

Download file with the first 29-convolutional layers of yolov4-tiny: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

```bash
   $ darknet.exe detector train darknet.data yolo-cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map
```

### Test the results

```bash
   $ darknet.exe detector test darknet.data yolo-cfg/yolov4-tiny-custom.cfg backup/yolov4-tiny-custom_final.weights data/your-image-path.png

```

### Run Inference in python

```bash
   $ cd src/
   $ python main.py --image your_image.png

```
