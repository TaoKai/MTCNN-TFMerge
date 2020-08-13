import os, sys
import cv2
import numpy as np
from codecs import open

def getRawData(basePath):
    rPath = 'adv_records.txt'
    lines = open(rPath, 'r', 'utf-8').read().strip().split('\n')
    records = []
    for l in lines:
        l_s = l.split(' ')
        p = basePath+l_s[0]
        c = l_s[1]
        b = [int(bb) for bb in l_s[2].split(',')]
        b[2] += b[0]
        b[3] += b[1]
        rec = {
            'file':p,
            'name':c,
            'box':b
        }
        records.append(rec)
    return records

def getNumpyData(rec):
    p = rec['file']
    b = rec['box']
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    b = np.array([b], dtype=np.float32)
    rec['image'] = img
    rec['box'] = b
    return rec

if __name__ == "__main__":
    basePath = 'E:/workspace/inveno_stars/广告视频/'
    recs = getRawData(basePath)
    for i, r in enumerate(recs):
        rc = getNumpyData(r)
        print(i, rc['image'].shape, rc['box'])