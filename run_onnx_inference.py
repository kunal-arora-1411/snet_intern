import onnxruntime
import numpy as np
import cv2
import os
import time
import psutil
import GPUtil
from tool.utils import load_class_names, post_processing, plot_boxes_cv2

# --- Config ---
onnx_model_path = "yolov4_608.onnx"
video_path = "/home/kunal/darknet/classroom.mp4"
class_names_path = "data/coco.names"
screenshot_dir = "screenshots"

os.makedirs(screenshot_dir, exist_ok=True)

# --- Load ONNX model ---
session = onnxruntime.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
IN_IMAGE_H, IN_IMAGE_W = input_shape[2], input_shape[3]
print("ONNX model loaded with input shape:", input_shape)

# --- Load video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# --- Output writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_onnx.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# --- Class labels ---
class_names = load_class_names(class_names_path)

# --- Benchmark tracking ---
frame_count = 0
total_inference_time = 0
start_total = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    pre_start = time.time()

    # Preprocess
    resized = cv2.resize(frame, (IN_IMAGE_W, IN_IMAGE_H))
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0) / 255.0

    # Inference
    inf_start = time.time()
    outputs = session.run(None, {input_name: img_in})
    inf_end = time.time()
    inference_time = (inf_end - inf_start) * 1000  # ms
    total_inference_time += inference_time

    # Post-processing
    boxes = post_processing(img_in, 0.4, 0.6, outputs)
    frame = plot_boxes_cv2(frame, boxes[0], class_names=class_names)

    # System metrics
    cpu_usage = psutil.cpu_percent()
    try:
        gpu_load = GPUtil.getGPUs()[0].load * 100
    except:
        gpu_load = -1  # No GPU

    # Overlay metrics
    label = f"Frame: {frame_count}, Inference: {inference_time:.2f}ms, CPU: {cpu_usage:.1f}%, GPU: {gpu_load:.1f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Save screenshot every 100 frames
    if frame_count % 100 == 0:
        screenshot_path = os.path.join(screenshot_dir, f"screenshot_{frame_count}.jpg")
        cv2.imwrite(screenshot_path, frame)

    # Show + Save
    cv2.imshow("YOLOv4 ONNX Video", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Final Stats ---
cap.release()
out.release()
cv2.destroyAllWindows()
end_total = time.time()

avg_inf_time = total_inference_time / frame_count
fps = frame_count / (end_total - start_total)

print("\n--- Benchmark Summary ---")
print(f"Total Frames: {frame_count}")
print(f"Average Inference Time: {avg_inf_time:.2f} ms")
print(f"Estimated FPS: {fps:.2f}")

