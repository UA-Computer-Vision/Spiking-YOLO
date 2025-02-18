import os
import torch.multiprocessing as mp
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    mp.freeze_support()
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  
    

    # model =YOLO("snn_yolov8l.yaml").load('69M_best.pt')
    model = YOLO('ultralytics/cfg/models/v8/snn_yolov8.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/snn_yolov8.yaml').load('runs/detect/train14/weights/last.pt')

    print(model)

    torch.cuda.empty_cache()

    #train
    # model.train(data="coco.yaml",device=[7],epochs=100)  # train the model
    # model.train(data="coco128.yaml",device=[4],epochs=100)  # train the model
    model.train(data="VisDrone.yaml",device=[0],epochs=100, batch=8) # train the model

    #TEST
    # model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)