import cv2
import torch
from numpy import random
from utils.general import apply_classifier, non_max_suppression
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.torch_utils import load_classifier 


camera_device = '/dev/video0'
cap = cv2.VideoCapture(camera_device)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, 100)

''' @ ERROR
im0s = run.im0s
classify =classifier.classify
modelc = classifier.modelc
img = inference.img
'''

weights = 'yolov7.pt'
device = torch.device("cuda")
model = attempt_load(weights, map_location=device)
augment = torch.no_grad()

while cap.isOpened():
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    ret, frames = cap.read()
    if not ret:
        break
    #right lens not use
    right_frame, left_frame = cv2.split(frames)
    #left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BAYER_GB2BGR)

    # TODO
    im0s = cv2.cvtColor(left_frame, cv2.COLOR_BAYER_GB2BGR)  # numpy.ndarray
    img = letterbox(im0s)
    
    '''pred = model(img, augment = False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, ())'''

    modelc = load_classifier(name = 'resnet101', n = 2)
    modelc.load_state_dict(torch.load("_?.pth"))

    classify = False
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
        cv2.imshow('result', left_frame)
        if cv2.waitKey(1) == ord('q'):
            break


#cv2.destroyAllWindows()