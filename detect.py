# -*- coding: utf-8 -*-
import os
from PIL import Image
from ultralytics import YOLO
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# model = YOLO("/path/to/trained_model/.pt") # Template
# model = YOLO('runs/detect/train1/weights/best.pt') # Example Usage #1 (trainings get auto placed here)
model = YOLO('best.pt') # Example Usage #2

# model.to("cpu") # Force model to run on CPU (If GPU w/ CUDA not available)

# Get image
# image_path = os.path.expanduser("/path/to/image/.jpg") # Template
image_path = os.path.expanduser("~/Downloads/aerial.webp") # Example Usage
image = Image.open(image_path)
image = image.resize((640, 640))

# Ensure gpu memory cache is cleared
torch.cuda.empty_cache()

# Conduct inference on image
results = model([image])

for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    im_rgb.show()
    im_rgb.save('result.jpg')  # save to disk

