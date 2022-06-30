import cv2
import numpy
import time

# source: https://github.com/mathbarc/darknet-demo/blob/master/api/yolo.py

def load_labels(labels_path: str):
    with open(labels_path) as labels_file:
        labels = labels_file.read()
    labels = labels.split("\n")
    labels = list(map(lambda x: x.strip(), labels))
    labels = dict({k: v for k, v in enumerate(labels)})
    labels.update({-1: 'Unknown'})
    return labels


class DNN:
    ONNX = 1
    DARKNET = 2
    TENSORFLOW = 3
    TORCH = 4
    

    def __init__(self, netCfgPath: str, weightsPathOrType, labels:list):

        if(type(weightsPathOrType) == str):
            self._model = cv2.dnn.readNetFromDarknet(netCfgPath, weightsPathOrType)
            self._type = DNN.DARKNET
        else:
            if weightsPathOrType == DNN.ONNX:
                self._model = cv2.dnn.readNetFromONNX(netCfgPath)
            elif weightsPathOrType == DNN.TENSORFLOW:
                self._model = cv2.dnn.readNetFromTensorflow(netCfgPath)
            elif weightsPathOrType == DNN.TORCH:
                self._model = cv2.dnn.readNetFromTorch(netCfgPath)
            else:
                raise "Invalid DNN Type"

            self._type = weightsPathOrType
        self._labels = labels
        

    def set_backend(self, backend:int, target:int):
        self._model.setPreferableBackend(backend)
        self._model.setPreferableTarget(target)

    def get_layers_names(self):
        layers_names = self._model.getLayerNames()
        output_layers = self._model.getUnconnectedOutLayers()
        output_layer_names = []
        for i in output_layers:
            output_layer_names.append(layers_names[i-1])
        return output_layer_names

    def inference(self, inputs) -> numpy.array:
        pass


class Yolo(DNN):
    
    def __init__(self, netCfgPath: str, weightsPathOrType, labels:list, size:list, 
                 conf_threshold:float=0.5, nms_threshold:float=0.3, 
                 scale:float = 1./255., swapRB:bool = True):
        DNN.__init__(self, netCfgPath, weightsPathOrType, labels)
        self._size = size
        self._swapRB = swapRB
        self._scale = scale
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._output_layers = self.get_layers_names()


    def inference(self, inputs)-> numpy.array:
        if not isinstance(inputs, numpy.ndarray) and len(inputs) == 1:
            inputs = inputs.pop()
        
        blob = cv2.dnn.blobFromImage(inputs, self._scale, self._size,
                                     swapRB=self._swapRB, crop=False)
        self._model.setInput(blob)
        layerOutputs = self._model.forward(self._output_layers)

        Width = inputs.shape[1]
        Height = inputs.shape[0]

        class_ids = []
        confidences = []
        boxes = []
        detections = []
        for out in layerOutputs:
            for detection in out:
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                confidence = scores[class_id]
                if confidence > self._conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self._conf_threshold,
                                   self._nms_threshold)

        if len(indices) > 0:
            for i in range(len(indices)):
                detection = {}
                detection["confidence"] = confidences[i]
                detection["label"] = self._labels[class_ids[i]]
                detection["roi"] = boxes[i]

                detections.append(detection)
        
        return detections
