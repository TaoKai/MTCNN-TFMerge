#711,348,65,69
import tensorflow as tf
import numpy as np
import cv2
from detect_face import PNet, RNet, ONet
import random
from get_adv_data import getRawData, getNumpyData

def get_scales_tf(img):
    minsize = tf.constant(20.0)
    factor = tf.constant(0.709)
    i0 = tf.constant(0)
    minl = tf.cast(tf.math.reduce_min(tf.shape(img)[:2]), tf.float32)
    m = 12.0/minsize
    minl = minl*m
    scales = tf.ones([1,])
    c = lambda i, scales, minl: minl>=12
    def body(i, scales, minl):
        pi = tf.cast(i, tf.float32)
        s = m*tf.math.pow(factor, pi)
        s_ex = tf.expand_dims(s, 0)
        scales = tf.concat([scales, s_ex], axis=0)
        minl = minl*factor
        i += 1
        return i, scales, minl
    ret = tf.while_loop(c, body, loop_vars=[i0, scales, minl], shape_invariants=[i0.get_shape(), tf.TensorShape([None,]), minl.get_shape()])
    return ret[1][1:]

def generateTFBox(imap, reg, scale, training):
    stride=tf.constant(2, dtype=tf.float32)
    cellsize=tf.constant(12, dtype=tf.float32)
    imap = tf.transpose(imap)
    dx1 = tf.transpose(reg[:,:,0])
    dy1 = tf.transpose(reg[:,:,1])
    dx2 = tf.transpose(reg[:,:,2])
    dy2 = tf.transpose(reg[:,:,3])
    if training:
        ind = tf.where(imap >= 0.0)
    else:
        ind = tf.where(imap >= 0.6)
    # y = ind[:, 0]
    # x = ind[:, 1]
    # if y.shape[0]==1:
    #     dx1 = tf.image.flip_up_down(dx1)
    #     dy1 = tf.image.flip_up_down(dy1)
    #     dx2 = tf.image.flip_up_down(dx2)
    #     dy2 = tf.image.flip_up_down(dy2)
    score = tf.gather_nd(imap, ind)
    dx1 = tf.gather_nd(dx1, ind)
    dy1 = tf.gather_nd(dy1, ind)
    dx2 = tf.gather_nd(dx2, ind)
    dy2 = tf.gather_nd(dy2, ind)
    reg = tf.stack([dx1, dy1, dx2, dy2], axis=0)
    reg = tf.transpose(reg)
    ind = tf.dtypes.cast(ind, tf.float32)
    q1 = tf.math.round((stride*ind+1)/scale)
    q2 = tf.math.round((stride*ind+cellsize-1+1)/scale)
    score = tf.transpose(tf.expand_dims(score, 0))
    boxes = tf.concat([q1, q2, score, reg], axis=1)
    return boxes

def computeIOU(pred_boxes, gt_boxes):
    g_len = tf.shape(gt_boxes)[0]
    p_len = tf.shape(pred_boxes)[0]
    px0 = pred_boxes[:, 0]
    py0 = pred_boxes[:, 1]
    px1 = pred_boxes[:, 2]
    py1 = pred_boxes[:, 3]
    gx0 = gt_boxes[:, 0]
    gy0 = gt_boxes[:, 1]
    gx1 = gt_boxes[:, 2]
    gy1 = gt_boxes[:, 3]
    px1 = tf.where(px1>px0, px1, px0)
    py1 = tf.where(py1>py0, py1, py0)
    px0 = tf.reshape(px0, (-1, 1))
    py0 = tf.reshape(py0, (-1, 1))
    px1 = tf.reshape(px1, (-1, 1))
    py1 = tf.reshape(py1, (-1, 1))
    gx0 = tf.reshape(gx0, (1, -1))
    gy0 = tf.reshape(gy0, (1, -1))
    gx1 = tf.reshape(gx1, (1, -1))
    gy1 = tf.reshape(gy1, (1, -1))
    px0mat = tf.tile(px0, (1, g_len))
    py0mat = tf.tile(py0, (1, g_len))
    px1mat = tf.tile(px1, (1, g_len))
    py1mat = tf.tile(py1, (1, g_len))
    gx0mat = tf.tile(gx0, (p_len, 1))
    gy0mat = tf.tile(gy0, (p_len, 1))
    gx1mat = tf.tile(gx1, (p_len, 1))
    gy1mat = tf.tile(gy1, (p_len, 1))
    cx0mat = tf.where(px0mat>gx0mat, px0mat, gx0mat)
    cy0mat = tf.where(py0mat>gy0mat, py0mat, gy0mat)
    cx1mat = tf.where(px1mat<gx1mat, px1mat, gx1mat)
    cy1mat = tf.where(py1mat<gy1mat, py1mat, gy1mat)
    cx1mat = tf.where(cx1mat>cx0mat, cx1mat, cx0mat)
    cy1mat = tf.where(cy1mat>cy0mat, cy1mat, cy0mat)
    parea = (px1mat-px0mat)*(py1mat-py0mat)
    garea = (gx1mat-gx0mat)*(gy1mat-gy0mat)
    carea = (cx1mat-cx0mat)*(cy1mat-cy0mat)
    iou = carea/(parea+garea-carea)
    iou_ind = tf.argmax(iou, axis=1)
    iou_sc = tf.reduce_max(iou, axis=1)
    return iou_ind, iou_sc


def build_pnet(training=True):
    with tf.variable_scope('pnet'):
        image_data = tf.placeholder(tf.float32, (None,None,3), 'input')
        gt_boxes = tf.placeholder(tf.float32, (None, 4), 'gt_box')
        scales = get_scales_tf(image_data)
        scale_len = tf.shape(scales)[0]
        #loop init
        total_boxes = tf.zeros([1, 9], dtype=tf.float32, name='total_boxes')
        i0 = tf.constant(0)
        c = lambda i, boxes: i < scale_len
        img_shp = tf.shape(image_data)
        img_shp = tf.dtypes.cast(img_shp, tf.float32)
        h_img = img_shp[0]
        w_img = img_shp[1]
        def body(i, boxes):
            scale = scales[i]
            hs = tf.math.round(h_img*scale)
            ws = tf.math.round(w_img*scale)
            im_data = tf.image.resize(image_data, (hs, ws), method='area')
            im_data = (im_data-127.5)*0.0078125
            img_x = tf.expand_dims(im_data, 0)
            img_y = tf.transpose(img_x, (0,2,1,3))
            pnet = PNet({'data':img_y})
            pout = [pnet.layers['conv4-2'], pnet.layers['prob1']]
            out0 = tf.transpose(pout[0], (0,2,1,3))
            out1 = tf.transpose(pout[1], (0,2,1,3))
            imap = out1[0,:,:,1]
            reg = out0[0,:,:,:]
            boxes0 = generateTFBox(imap, reg, scale, training=training)
            if training:
                pick_boxes = boxes0
            else:
                x1 = boxes0[:, 0]
                y1 = boxes0[:, 1]
                x2 = boxes0[:, 2]
                y2 = boxes0[:, 3]
                scores = boxes0[:, 4]
                boxes_nms = tf.transpose(tf.stack([y1, x1, y2, x2], axis=0))
                nms_inds = tf.image.non_max_suppression(boxes_nms, scores, 2000, iou_threshold=0.5)
                pick_boxes = tf.gather(boxes0, nms_inds)
            boxes = tf.concat([boxes, pick_boxes], axis=0)
            boxes = tf.reshape(boxes, [-1, 9])
            i += 1
            return i, boxes
        ret = tf.while_loop(c, body, loop_vars=[i0, total_boxes], shape_invariants=[i0.get_shape(), tf.TensorShape([None, 9])])
        pick_boxes1 = ret[1][1:, :]
        pick_boxes2 = pick_boxes1
        if not training:
            input_boxes = pick_boxes1
            x1 = input_boxes[:, 0]
            y1 = input_boxes[:, 1]
            x2 = input_boxes[:, 2]
            y2 = input_boxes[:, 3]
            scores = input_boxes[:, 4]
            boxes_nms1 = tf.transpose(tf.stack([y1, x1, y2, x2], axis=0))
            nms_inds = tf.image.non_max_suppression(boxes_nms1, scores, 2000, iou_threshold=0.7)
            pick_boxes2 = tf.gather(input_boxes, nms_inds)
        regw = pick_boxes2[:,2]-pick_boxes2[:,0]
        regh = pick_boxes2[:,3]-pick_boxes2[:,1]
        qq1 = pick_boxes2[:,0]+pick_boxes2[:,5]*regw
        qq2 = pick_boxes2[:,1]+pick_boxes2[:,6]*regh
        qq3 = pick_boxes2[:,2]+pick_boxes2[:,7]*regw
        qq4 = pick_boxes2[:,3]+pick_boxes2[:,8]*regh
        pick_boxes3 = tf.transpose(tf.stack([qq1, qq2, qq3, qq4, pick_boxes2[:,4]], axis=0))
        iou_ind, iou_sc = computeIOU(pick_boxes3, gt_boxes)
        tie_boxes = tf.gather(gt_boxes, iou_ind)
        iou04ind = tf.reshape(tf.where(iou_sc>0.4), (-1,)) #which score is right?
        len04 = tf.shape(iou04ind)[0]
        judge = len04>0
        iou04ind = tf.cond(judge, lambda:iou04ind, lambda:getTopIndices(iou_sc))
        boxes04 = tf.gather(pick_boxes3, iou04ind)[:,:4]
        boxesGt = tf.gather(tie_boxes, iou04ind)
        scores = pick_boxes3[:, 4]
        iou03ind = getIouInd03(scores, iou_sc, len04)
        boxes03 = tf.gather(pick_boxes3, iou03ind)
        iou06ind = tf.reshape(tf.where(iou_sc>0.65), (-1,))
        boxes06 = tf.gather(pick_boxes3, iou06ind)
        nag03 = boxes03[:, 4]
        nagGt = tf.zeros(tf.shape(nag03))
        loss03 = tf.nn.sigmoid_cross_entropy_with_logits(labels=nagGt, logits=nag03)
        loss03 = tf.reduce_mean(loss03)
        pos06 = boxes06[:, 4]
        posGt = tf.ones(tf.shape(pos06))
        loss06 = tf.nn.sigmoid_cross_entropy_with_logits(labels=posGt, logits=pos06)
        loss06 = tf.reduce_mean(loss06)
        loss06 = tf.cond(tf.is_nan(loss06), lambda:tf.constant(0, dtype=tf.float32), lambda: loss06)
        loss04 = getBoxLoss04(boxes04, boxesGt)
        loss346 = loss03+loss04+loss06
        optim = tf.train.AdamOptimizer(learning_rate=0.003)
        train_opt = optim.minimize(loss346)
        picks = getSomeBoxes(pick_boxes3)
    return train_opt, loss346, image_data, gt_boxes, picks

def getSomeBoxes(boxes):
    scores = boxes[:, 4]
    topInd = tf.math.top_k(scores, k=30, sorted=False).indices
    pick_boxes = tf.gather(boxes, topInd)
    return pick_boxes

def getBoxLoss04(boxes04, boxesGt):
    # gt_w = boxesGt[:, 2]-boxesGt[:, 0]
    # gt_h = boxesGt[:, 3]-boxesGt[:, 1]
    x0_diff = boxes04[:, 0]-boxesGt[:, 0]
    y0_diff = boxes04[:, 1]-boxesGt[:, 1]
    x1_diff = boxes04[:, 2]-boxesGt[:, 2]
    y1_diff = boxes04[:, 3]-boxesGt[:, 3]
    # x0_diff = x0_diff/gt_w
    # x1_diff = x1_diff/gt_w
    # y0_diff = y0_diff/gt_h
    # y1_diff = y1_diff/gt_h
    square = tf.sqrt(tf.pow(x0_diff, 2)+tf.pow(y0_diff, 2))+tf.sqrt(tf.pow(x1_diff, 2)+tf.pow(y1_diff, 2))
    loss = tf.reduce_sum(0.5*square)*0.001
    return loss

def getIouInd03(scores, iou_sc, len04):
    judge = len04>0
    ind03 = tf.reshape(tf.where(iou_sc<0.3), (-1,))
    score03 = tf.gather(scores, ind03)
    k = tf.cond(judge, lambda:len04*3, lambda:50)
    top03ind = tf.math.top_k(score03, k=k, sorted=False).indices
    iou03ind = tf.gather(ind03, top03ind)
    return iou03ind

def getTopIndices(iou_sc):
    iou0ind = tf.reshape(tf.where(iou_sc>=0), (-1,))
    iou0sc = tf.gather(iou_sc, iou0ind)
    len0 = tf.shape(iou0sc)[0]
    judge = (len0>=10)
    k = tf.cond(judge, lambda:tf.constant(10), lambda:len0)
    topInd = tf.math.top_k(iou0sc, k=k, sorted=False).indices
    iou0ind = tf.gather(iou0ind, topInd)
    return iou0ind

def draw(img, boxes):
    boxes = list(boxes)
    random.shuffle(boxes)
    for b in boxes:
        b0 = int(b[0])
        b1 = int(b[1])
        b2 = int(b[2])
        b3 = int(b[3])
        score = b[4]
        cv2.rectangle(img, (b0, b1), (b2, b3), (0,0,255), 1)
        print('drawing', b)
        cv2.imshow('img', img)
    cv2.waitKey(100)

def train(epoch=20):
    basePath = 'E:/workspace/inveno_stars/广告视频/'
    recs = getRawData(basePath)
    random.shuffle(recs)
    sess = tf.Session()
    opt, loss, image_data, gt_boxes, picks = build_pnet(training=True)
    sess.run(tf.global_variables_initializer())
    for ep in range(epoch):
        for i, rec in enumerate(recs):
            r = getNumpyData(rec)
            cost, _ = sess.run([loss, opt], {image_data:r['image'], gt_boxes:r['box']})
            print('epoch', ep, i, cost)

if __name__ == "__main__":
    train(epoch=20)
