import tensorflow as tf
import os, sys
import numpy as np
import cv2
import detect_face
import random
from skimage import transform as trans
import shutil


def loadModel():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    g = tf.get_default_graph().as_graph_def()
    for n in g.node:
        print(n.name)
    return pnet, rnet, onet

def detect(img, nets):
    shp = img.shape[:2]
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    boxes, points = detect_face.detect_face(img, minsize, nets[0], nets[1], nets[2], threshold, factor)
    boxes[boxes<0] = 0
    boxes[boxes[:, 0]>shp[1]] = shp[1]
    boxes[boxes[:, 2]>shp[1]] = shp[1]
    boxes[boxes[:, 1]>shp[0]] = shp[0]
    boxes[boxes[:, 3]>shp[0]] = shp[0]
    points[points<0] = 0
    points = points.T
    return list(boxes), list(points)

def faceAlign(img, points):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
    dst = np.array(points, dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    shp = [112, 96]
    if M is None:
        return None
    else:
        warped = cv2.warpAffine(img, M, (shp[1],shp[0]), borderValue = 0.0)
        return warped
    

def drawFaces(img, boxes, points, savePath):
    def cutBox(img, box):
        b=box.astype(np.int32)
        cutImg = img[b[1]:b[3]+1, b[0]:b[2]+1, :]
        cutImg = cv2.resize(cutImg, (96, 112))
        return cutImg

    def batchShow(cuts):
        col_num = int(np.sqrt(len(cuts)))+1
        h = len(cuts)//col_num+1
        if len(cuts)%col_num==0:
            h -= 1
        img_b = np.zeros([112*h, 96*col_num, 3], dtype=np.uint8)
        for i,c in enumerate(cuts):
            w = i%col_num
            h = i//col_num
            img_b[h*112:(h+1)*112, w*96:(w+1)*96, :] = c
        return img_b

    cuts = []
    orig_img = img.copy()
    for i,b in enumerate(boxes):
        if int(b[0])>=int(b[2]) or int(b[1])>=int(b[3]):
            continue
        # cut = cutBox(orig_img, b)   
        b0 = int(b[0])
        b1 = int(b[1])
        b2 = int(b[2])
        b3 = int(b[3])
        score = b[4]
        if score < 0.98:
            continue
        mid = (int((b0 + b2) / 2) - 15, int((b1 + b3) / 2)+5)
        top = (int((b0 + b2) / 2) - 15, b1-4)
        cv2.rectangle(img, (b0, b1), (b2, b3), (0,0,255), 2)
        cv2.putText(img, str(score*100)[:4], top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        pts = points[i].astype(np.int32)
        ptsX = pts[:5]
        ptsY = pts[5:]
        ptsXY = np.array([ptsX, ptsY], dtype=np.float32).T
        cut = faceAlign(orig_img, ptsXY)
        if cut is None:
            cut = cutBox(orig_img, b)
        cuts.append(cut)
        if savePath is not None:
            cv2.imwrite(os.path.join(savePath, str(i)+'.jpg'), cut)
        for i in range(ptsX.shape[0]):
            cv2.circle(img, (ptsX[i], ptsY[i]), 2, (0,0,255))
    cv2.imshow('image', img)
    if len(cuts)>0:
        batch_img = batchShow(cuts)
        cv2.imshow('faces', batch_img)
    return cv2.waitKey(0)


def readStars(path):
    dirs = os.listdir(path)
    dirs.sort()
    ids={}
    imgs={}
    for i, d in enumerate(dirs):
        ids[i] = d
        imgs[i] = []
        d = os.path.join(path, d)
        picDirs = os.listdir(d)
        for p in picDirs:
            imgs[i].append(os.path.join(d, p))
    return ids, imgs

def saveCuts(img, boxes, points, savePath, picId):
    def cutBox(img, box):
        b=box.astype(np.int32)
        cutImg = img[b[1]:b[3]+1, b[0]:b[2]+1, :]
        cutImg = cv2.resize(cutImg, (96, 112))
        return cutImg
    for i, b in enumerate(boxes):
        b0 = int(b[0])
        b1 = int(b[1])
        b2 = int(b[2])
        b3 = int(b[3])
        score = b[4]
        if score < 0.98:
            continue
        pts = points[i].astype(np.int32)
        ptsX = pts[:5]
        ptsY = pts[5:]
        ptsXY = np.array([ptsX, ptsY], dtype=np.float32).T
        cut = faceAlign(img, ptsXY)
        if cut is None:
            cut = cutBox(img, b)
        # cut = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(savePath, str(picId)+'_'+str(i)+'.jpg'), cut, [int(cv2.IMWRITE_JPEG_QUALITY),100])


def cutFaces(datasetPath, ids, imgs, func):
    dirs = list(ids.keys())
    dirs.sort()
    for d in dirs:
        path = os.path.join(datasetPath, str(d))
        if not os.path.exists(path):
            os.makedirs(path)
    if os.path.exists(os.path.join(datasetPath, 'id_name.txt')):
        os.remove(os.path.join(datasetPath, 'id_name.txt'))
    f = open(os.path.join(datasetPath, 'id_name.txt'), 'a')
    for k, v in ids.items():
        f.write(str(k)+' '+v+'\n')
    f.close()
    cnt = 0
    for k, imList in imgs.items():
        cnt += 1
        if cnt<=0:
            continue
        savePath = os.path.join(datasetPath, str(k))
        random.shuffle(imList)
        for i, im in enumerate(imList[:200]):
            if '.jpg' not in im.lower() and '.jpeg' not in im.lower():
                continue
            try:
                img = cv2.imdecode(np.fromfile(im, dtype=np.uint8),-1)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                boxes, points = detect(img, func)
                saveCuts(img, boxes, points, savePath, i)
                print('detect', im)
            except:
                print('read error', im)




if __name__ == "__main__":
    imgPath = "stars.jpg"
    savePath = "D:\\workspace\\data\\mtcnn_faces\\99"
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # shp = (img.shape[1], img.shape[0])
    # M = cv2.getRotationMatrix2D((shp[0]//2, shp[1]//2), 45, 1.0)
    # rot = cv2.warpAffine(img, M, shp)
    nets = loadModel()
    boxes, points=detect(img_cvt, nets)
    drawFaces(img, boxes, points, None)
    # starsPath = "E:\\BaiduNetdiskDownload\\seeprettyface_chs_stars_original\\seeprettyface_chs_stars_original\\chs_stars_original"
    # datasetPath = "E:\\BaiduNetdiskDownload\\seeprettyface_chs_stars_original\\seeprettyface_chs_stars_original\\chs_stars_faces"
    # ids, imgs = readStars(starsPath)
    # cutFaces(datasetPath, ids, imgs, nets)
