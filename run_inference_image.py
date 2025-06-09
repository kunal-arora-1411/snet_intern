import onnxruntime
import numpy as np
import cv2
from tool.utils import load_class_names, post_processing, plot_boxes_cv2

# Config
onnx_model_path = "yolov4_608.onnx"
image_path = "dog.jpg"
class_names_path = "data/coco.names"

# Load model
session = onnxruntime.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
IN_IMAGE_H, IN_IMAGE_W = input_shape[2], input_shape[3]
print("ONNX model loaded with input shape:", input_shape)

# Preprocess
image_src = cv2.imread(image_path)
resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H))
img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
img_in = np.expand_dims(img_in, axis=0) / 255.0

# Inference
outputs = session.run(None, {input_name: img_in})

# Post-process and plot
boxes = post_processing(img_in, 0.4, 0.6, outputs)
class_names = load_class_names(class_names_path)
plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)
