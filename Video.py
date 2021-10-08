import os
import numpy as np
import pandas as pd
# import face_recognition
import datetime
import csv
import sys
import cv2
import copy
import multiprocessing
from multiprocessing import Pool
from collections import Counter


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # For CPU inference


sys.path.append(os.path.join(sys.path[0], 'nomeroff-net'))

from NomeroffNet.YoloV5Detector import Detector
detector = Detector()                                   #
detector.load()                                         #

from NomeroffNet.BBoxNpPoints import NpPointsCraft


from NomeroffNet.tools.image_processing import getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
npPointsCraft = NpPointsCraft()
npPointsCraft.load()

from NomeroffNet.OptionsDetector import OptionsDetector
optionsDetector = OptionsDetector()
optionsDetector.load("latest")

from NomeroffNet.TextDetector import TextDetector
# Initialize text detector.
textDetector = TextDetector({
       "eu_ua_2004_2015": {
           "for_regions": ["eu_ua_2015", "eu_ua_2004"],
           "model_path": "latest"
       },
       "eu_ua_1995": {
           "for_regions": ["eu_ua_1995"],
           "model_path": "latest"
       },
       "eu": {
           "for_regions": ["eu"],
           "model_path": "latest"
       },
       "ru": {
           "for_regions": ["ru", "eu-ua-fake-lnr", "eu-ua-fake-dnr"],
           "model_path": "latest"
       },
       "kz": {
           "for_regions": ["kz"],
           "model_path": "latest"
       },
       "ge": {
           "for_regions": ["ge"],
           "model_path": "latest"
       },
       "su": {
           "for_regions": ["su"],
           "model_path": "latest"
       }
   })


camera_str = 'videoplayback (1).mp4' # camera or video
cap = cv2.VideoCapture(camera_str)
cv2.namedWindow('Video output')    #
cv2.startWindowThread()


while cap.isOpened():
    ret, frame = cap.read()
    if frame is None or not cap.isOpened():
        print('End or error')
        break  # если нет кадра или подключения, прерываемся

    N = 10
    H = 1.3
    for i in range(N):  # skip N frames
        ret, frame = cap.read()  # для чего мы их получали, хз, возможно это задержка

    frame = cv2.resize(frame, (int(frame.shape[1] / H), int(frame.shape[0] / H)))

    rgb_frame = frame[:, :, ::-1]

    targetBoxes = detector.detect_bbox(copy.deepcopy(rgb_frame))
    targetBoxes = targetBoxes  # why?

    all_points = npPointsCraft.detect(rgb_frame, targetBoxes)
    all_points = [ps for ps in all_points if len(ps)]
    # print(all_points)
    # cut zones
    toShowZones = [getCvZoneRGB(rgb_frame, reshapePoints(rect, 1)) for rect in all_points]
    zones = convertCvZonesRGBtoBGR(toShowZones)

    '''
    for zone, points in zip(toShowZones, all_points):
        cv2.imshow("zone", zone[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
    '''

    # find standart

    # regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionIds, countLines = optionsDetector.predict(zones)

    regionNames = optionsDetector.getRegionLabels(regionIds)
    # print(regionNames)
    # print(countLines)

    # find text with postprocessing by standart
    textArr = textDetector.predict(zones, regionNames, countLines)

    # draw rect and 4 points
    for targetBox, points in zip(targetBoxes, all_points):
        rgb_frame = np.array(rgb_frame)
        cv2.rectangle(rgb_frame,
                      (int(targetBox[0]), int(targetBox[1])),
                      (int(targetBox[2]), int(targetBox[3])),
                      (0, 120, 255),
                      3)

        font = cv2.FONT_HERSHEY_DUPLEX

    cv2.imshow('Video output', rgb_frame[:, :, ::-1])
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break


# cap.release()
cv2.destroyAllWindows()
