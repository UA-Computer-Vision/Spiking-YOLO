# -*- coding: utf-8 -*-
import os
from PIL import Image
from ultralytics import YOLO

# model = YOLO("/path/to/checkpoint/.pt")
model = YOLO('runs/detect/train14/weights/best.pt')
# results = model(['/path/to/data/.jpg'])
# results = model(['D:/Downloaded Pictures/uruguay costumes.jpg'])
# results = model(['D:/Downloaded Pictures/Aerial-view-over-Punta-Del-Este-and-Atlantic-Ocean-at-sunset-Uruguay_647676238.jpg'])
# results = model(['D:/Downloaded Pictures/blackwhite miate.jpg'])
results = model(['C:/lxh/lxh_data/coco128/images/train2017/000000000312.jpg'])
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    im_rgb.show()
    im_rgb.save('result.jpg')  # save to disk

