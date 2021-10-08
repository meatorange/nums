import os
import numpy as np

import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

camera_str = 'videoplayback (1).mp4' # camera or video
cap = cv2.VideoCapture(camera_str)
cv2.namedWindow('Video output')    #
cv2.startWindowThread()

while cap.isOpened():
    for i in range(10):  # Получаем 120 кадров (?)
        ret, frame = cap.read()  # для чего мы их получали, хз, возможно это задержка

    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1] / 1.3), int(frame.shape[0] / 1.3)))
    # frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))

    if frame is None or not cap.isOpened():
        print('End or error')
        break  # если нет кадра или подключения, прерываемся

    rgb_frame = frame[:, :, ::-1]

    cv2.imshow('Video output', rgb_frame[:, :, ::-1])
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break


cap.release()
cv2.destroyAllWindows()

