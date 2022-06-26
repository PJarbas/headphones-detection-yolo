

## Headphones object detection using YoloV4

This project aims to train a object detecion deep learning model using custom number of classes (Headphones)


### Get started

Ensure that your **python** version is >= 3.9

Download the data running the following command in the `src` directory:

```bash
   $ python data_downloader.py 
```


Create the `train.txt` and `test.txt` files with random split, these files must be in the same directory that the `darknet.data` file

```bash
   $ cd src
   $ python split_train_test.py ../data/images/ 
```


### Install YoloV4 Darknet:

* https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg

* https://medium.com/geekculture/yolov4-darknet-installation-and-usage-on-your-system-windows-linux-8dec2cea6e81

### Download the yolov4-tiny

Get the yolov4-tiny.cfg and change the number of classes and the filters for the [conv] layers before the [yolo] layers 

```bash
   $ wget https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg
```

### Train

```bash
   $ darknet detector train darknet.data yolo-cfg/yolov4-tiny.cfg  
```



