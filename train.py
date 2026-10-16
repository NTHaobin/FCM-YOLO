from mpmath import workdps
from sympy import false
from ultralytics import YOLO
from ultralytics import settings


# 强制更新设置，确保 TensorBoard 开启
settings.update({'tensorboard': False})

# import warnings
# warnings.filterwarnings('ignore')

if __name__ == '__main__':
  model = YOLO('FCM-YOLO11.yaml')

# Train the model
  results = model.train(
    data=r'D:\ntpy\yolo\2yolo11\ultralytics-main\ultralytics\cfg\datasets\VisDrone.yaml',
    epochs=300,
    lr0=0.01,
    lrf=0.0001,
    batch=8,
    # accumulate=2,
    imgsz=640,
    # amp=True,
    workers=0,
    # mosaic=1.0,
    # mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    optimizer='SGD',
    momentum=0.937,
    # copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
    device="0",
    close_mosaic=10,
    resume=False,
    # verbose=False,
  )


