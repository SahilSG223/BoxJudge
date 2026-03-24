from ultralytics import YOLO
import torch
import torchvision
import torchvision.transforms as transforms


model = YOLO("yolov8m.pt")