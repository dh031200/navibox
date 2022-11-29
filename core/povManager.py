import cv2
import numpy as np


class POVManager:
    def __init__(self, cs, ws):
        cs = np.float32(cs)
        ws = np.float32(ws)

        self.h = cv2.findHomography(ws, cs, cv2.RANSAC, 5.0)[0]

    def coord_transform(self, obj):
        x, y = obj.xywh[:2]
        after = np.matmul(self.h, np.array([x, y, 1]).transpose())
        after = after / after[2]
        after = after[:2]
        after = np.round(after, 0).astype(int).transpose()

        return after
