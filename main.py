import os
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from IPython.display import display, Image

model = YOLO("yolov10n.pt")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Define dataset paths
dataset_path = "./dataset"  
train_path = os.path.join("E:\Liscence Plate Recognition\dataset\train", "train")
val_path = os.path.join("E:\Liscence Plate Recognition\dataset\val", "val")
test_path = os.path.join("E:\Liscence Plate Recognition\dataset\test", "test")

dataset_yaml_path = "E:/Liscence Plate Recognition/dataset/dataset.yaml"
model.train(
    data=dataset_yaml_path,
    epochs=5,
    batch=8,
    imgsz=640,
    device=device,
    name="license_plate_detector"
)
# Load the trained model
trained_model = YOLO("runs/detect/license_plate_detector/weights/best.pt")

# Function to run the model on images
def detect_license_plate(image_path):
    results = trained_model(image_path)  # Run inference
    for result in results:
        img = result.plot()  # Draw bounding boxes
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Test with an image
test_image = os.path.join(test_path,) 
detect_license_plate(test_image)

# Function to run inference on a video
def detect_license_plate_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    plt.ion()  # Enable interactive mode for live updates
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = trained_model(frame)
        for result in results:
            frame = result.plot()

        out.write(frame)
        
        # Display the frame using Matplotlib instead of cv2.imshow
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)
        plt.clf()
    
    cap.release()
    out.release()
    plt.ioff()  # Disable interactive mode
    plt.close()

# Test with a video
detect_license_plate_video("E:/Liscence Plate Recognition/cars2.mp4", "E:/Liscence Plate Recognition/output_video/output_car.mp4")

