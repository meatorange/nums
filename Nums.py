import os
import numpy as np
import pandas as pd
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


# NomeroffNet path
# DIR = ''
# NOMEROFF_NET_DIR = os.path.abspath(DIR)   # TODO: change dir
# sys.path.append(NOMEROFF_NET_DIR)

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

# на этом моменте все уже загружено и настроено, теперь можно получить кадр и обработать его 

frame = cv2.imread('3.jpg')
# frame = cv2.imread('2.png')
# frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
# frame = cv2.cvtColor(frame, cv2 .COLOR_BGR2GRAY)   # попробуем перевести в серый, мб в это была проблема, зависит от камеры

cv2.imshow("Auto", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


rgb_frame = frame[:, :, ::-1]

targetBoxes = detector.detect_bbox(copy.deepcopy(rgb_frame))
targetBoxes = targetBoxes   # why?

all_points = npPointsCraft.detect(rgb_frame, targetBoxes)
all_points = [ps for ps in all_points if len(ps)]
# print(all_points)
# cut zones
toShowZones = [getCvZoneRGB(rgb_frame, reshapePoints(rect, 1)) for rect in all_points]
zones = convertCvZonesRGBtoBGR(toShowZones)

# find standart

# regionIds, stateIds, countLines = optionsDetector.predict(zones)
regionIds, countLines = optionsDetector.predict(zones)

regionNames = optionsDetector.getRegionLabels(regionIds)
# print(regionNames)
# print(countLines)

# find text with postprocessing by standart
textArr = textDetector.predict(zones, regionNames, countLines)
# textArr = textPostprocessing(textArr, regionNames)

textArr = [text for text in textArr if text != '']

if len(textArr) != 0:
    for numberplate in textArr:
        print(numberplate, '\n')
else:
    print('Nothing to print')
