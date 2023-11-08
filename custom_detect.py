import random
from traceback import print_exc

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import plot_one_box


# Global Variables
WEIGHT = 'yolov7.pt'
DEVICE = torch.device('cuda')
IS_HALF = DEVICE.type == 'cuda'
IS_AUGMENT = False


def get_capture(source:str='/dev/video0',
                width:int=640,
                height:int=480,
                fps:int=45,
                exposure:int=100) -> cv2.VideoCapture:
    ''' 비디오 캡쳐 객체를 가져온다. (oCamS-1CGN-U)
    ----- -----
    * 인자값:
        source (str): 카메라 장치 경로
    * 반환값:
        cap (cv2.VideoCapture): 비디오 캡쳐 객체
    * 예외:
        RuntimeError: 카메라 장치를 열지 못했을 때
    ----- ----- '''
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError('Failed to open ...')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    return cap


def read_frame(cap:cv2.VideoCapture) -> np.ndarray:
    ''' 스테레오 카메라의 왼쪽 프레임 이미지를 읽어온다. (oCamS-1CGN-U)
    ----- -----
    * 인자값:
        cap (cv2.VideoCapture): 비디오 캡쳐 객체
    * 반환값:
        l_frame (np.ndarray): 왼쪽 프레임 이미지
    * 예외:
        RuntimeError: 프레임을 읽어오지 못하였을 때
    ----- ----- '''

    ret, frames = cap.read()
    if not ret:
        raise RuntimeError('Failed to read ...')

    r_frame, l_frame = cv2.split(frames)  # right and left
    l_frame = cv2.cvtColor(l_frame, cv2.COLOR_BAYER_GB2BGR)

    return l_frame


def crop_center_square(img:np.ndarray) -> np.ndarray:
    ''' 입력 이미지의 정중앙을 원점으로 하는, 가장 큰 정사각형 이미지를 반환(크롭)한다.
        이 정사각형의 한 변의 길이는 입력 이미지의 높이와 너비 중 더 작은 것과 같다.
    ----- -----
    * 인자값:
        img (np.ndarray): 입력 이미지
    * 출력값:
        (np.ndarray): 크롭된 정사각형 이미지
    ----- ----- '''
    img_h, img_w, _ = img.shape  # height, width, and channels of the image
    side = min(img_h, img_w)

    gap_h = int((img_h - side) / 2)
    gap_w = int((img_w - side) / 2)

    return img[gap_h : gap_h + side, gap_w : gap_w + side, :]


def load_yolov7() -> torch.nn.Module:
    ''' Yolov7 객체 탐지 모델을 로드한다.
    ----- -----
    * 출력값:
        model (torch.nn.Module): ...
    ----- ----- '''
    model = attempt_load(WEIGHT, DEVICE)

    if DEVICE.type == 'cuda':
        model.half()

    return model


def proc(img:np.ndarray) -> np.ndarray:
    ''' 이미지를 모델에 입력할 수 있는 형태로 전처리한다.
    ----- -----
    * 인자값:
        img (np.ndarray): 입력 이미지
    * 출력값:
        img_ (np.ndarray): 전처리된 출력 이미지
    ----- ----- '''
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ = img_.transpose(2, 0, 1)
    img_ = np.ascontiguousarray(img_)
    img_ = torch.from_numpy(img_).to(DEVICE)
    img_ = img_.half() if IS_HALF else img_.float()
    img_ = img_ / 255.0
    if img_.ndimension() == 3:
        img_ = img_.unsqueeze(0)

    return img_


if __name__ == '__main__':
    camera_source = '/dev/video0'
    width, height, fps = 640, 480, 45
    # -----
    # 카메라 세팅
    cap = get_capture(camera_source, width, height, fps)
    delay_mm_sec = int(1 / fps * 1000)
    # -----
    # 객체 탐지 모델 로드
    model = load_yolov7()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    try:
        with torch.no_grad():
            while cap.isOpened():
                # -----
                # 프레임 전처리
                frame = read_frame(cap)
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (640, 640))
                # ----
                # 객체 탐지
                preds = model(proc(frame), IS_AUGMENT)[0]
                preds = non_max_suppression(preds)
                # -----
                # 탐지 결과 그리기(drawing)
                for i, pred in enumerate(preds):
                    if len(pred):
                        for *xyxy, conf, cls in reversed(pred):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy,
                                         frame,
                                         colors[int(cls)],
                                         label,
                                         line_thickness=1)
                # -----
                # 결과 화면 출력
                cv2.imshow('result', frame)
                if cv2.waitKey(delay_mm_sec) == ord('q'):
                    break
    except:
        print_exc()

    cv2.destroyAllWindows()
    cap.release()