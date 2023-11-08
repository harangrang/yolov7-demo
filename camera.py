import cv2

camera_device = '/dev/video0'
cap = cv2.VideoCapture(camera_device)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, 100)

while cap.isOpened():
    ret, frames = cap.read()
    if not ret:
        break
    #right lens not use
    right_frame, left_frame = cv2.split(frames)
    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BAYER_GB2BGR)

    #to do
    cv2.imshow('result', left_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()