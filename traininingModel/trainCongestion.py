from ultralytics import YOLO
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
model=YOLO("yolo11n.pt")

batch_size = 100
img_height = 250
img_width = 250

# Train the model
# train_results = model.train(
#     data=r"C:\Users\tar30\makeForLkoTraffic\traininingModel\data.yml",  # path to dataset YAML
#e
# )


model = YOLO(r"C:\Users\tar30\makeForLkoTraffic\Models\best.pt")

# Perform object detection on an image
results = model(r"C:\Users\tar30\Downloads\Wormhole WqMkML\test (14).mp4", save=True)
