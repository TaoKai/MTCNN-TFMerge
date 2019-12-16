import os, sys
import numpy as np
import cv2
from detect_face import PNet, RNet, ONet
import tensorflow as tf
from six import string_types, iteritems


def bbreg(boxes, reg):
    w = boxes[:,2]-boxes[:,0]+1
    h = boxes[:,3]-boxes[:,1]+1
    b1 = boxes[:,0]+reg[:,0]*w
    b2 = boxes[:,1]+reg[:,1]*h
    b3 = boxes[:,2]+reg[:,2]*w
    b4 = boxes[:,3]+reg[:,3]*h
    score = boxes[:, 4]
    boxes = tf.transpose(tf.stack([b1, b2, b3, b4, score], axis=0))
    return boxes


def rerec(boxes):
    h = boxes[:,3]-boxes[:,1]
    w = boxes[:,2]-boxes[:,0]
    l = tf.maximum(h, w)
    pb0 = boxes[:,0]+w*0.5-l*0.5
    pb1 = boxes[:,1]+h*0.5-l*0.5
    pb2 = pb0 + l
    pb3 = pb1 + l
    pb4 = boxes[:,4]
    boxes = tf.transpose(tf.stack([pb0, pb1, pb2, pb3, pb4], axis=0))
    return boxes


def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the va
        lue of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.

        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)


def simple_crop(boxes, w, h):
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    scores = boxes[:, 4]
    one_vec = tf.ones(tf.shape(x0), dtype=tf.float32)
    h_vec = tf.ones(tf.shape(x0), dtype=tf.float32)*h
    w_vec = tf.ones(tf.shape(x0), dtype=tf.float32)*w
    x0 = tf.where(x0<1, one_vec, x0)
    y0 = tf.where(y0<1, one_vec, y0)
    x1 = tf.where(x1<=x0, x0+1, x1)
    y1 = tf.where(y1<=y0, y0+1, y1)
    x1 = tf.where(x1>w, w_vec, x1)
    y1 = tf.where(y1>h, h_vec, y1)
    crop_boxes = tf.transpose(tf.stack([x0, y0, x1, y1, scores], axis=0))
    return crop_boxes


def generateTFBox(imap, reg, scale):
    stride=tf.constant(2, dtype=tf.float32)
    cellsize=tf.constant(12, dtype=tf.float32)
    imap = tf.transpose(imap)
    dx1 = tf.transpose(reg[:,:,0])
    dy1 = tf.transpose(reg[:,:,1])
    dx2 = tf.transpose(reg[:,:,2])
    dy2 = tf.transpose(reg[:,:,3])
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


def MyMtcnnNet(sess):
    def load_np(data_path, session, ignore_missing=False):
        """Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    model_path,_ = os.path.split(os.path.realpath(__file__))
    with tf.variable_scope('pnet'):
        image_data = tf.placeholder(tf.float32, (None,None,3), 'input')
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
            boxes0 = generateTFBox(imap, reg, scale)
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
        load_np(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet'):
        input_boxes = pick_boxes1
        x1 = input_boxes[:, 0]
        y1 = input_boxes[:, 1]
        x2 = input_boxes[:, 2]
        y2 = input_boxes[:, 3]
        scores = input_boxes[:, 4]
        boxes_nms1 = tf.transpose(tf.stack([y1, x1, y2, x2], axis=0))
        nms_inds = tf.image.non_max_suppression(boxes_nms1, scores, 2000, iou_threshold=0.7)
        pick_boxes = tf.gather(input_boxes, nms_inds)
        regw = pick_boxes[:,2]-pick_boxes[:,0]
        regh = pick_boxes[:,3]-pick_boxes[:,1]
        qq1 = pick_boxes[:,0]+pick_boxes[:,5]*regw
        qq2 = pick_boxes[:,1]+pick_boxes[:,6]*regh
        qq3 = pick_boxes[:,2]+pick_boxes[:,7]*regw
        qq4 = pick_boxes[:,3]+pick_boxes[:,8]*regh
        pick_boxes = tf.transpose(tf.stack([qq1, qq2, qq3, qq4, pick_boxes[:,4]], axis=0))
        pick_boxes2 = rerec(pick_boxes)
        pick_boxes2 = simple_crop(pick_boxes2, w_img, h_img) # x, y
        crop_boxes = transform_fpcoor_for_tf(pick_boxes2[:, :4], img_shp, [24, 24])
        # box_ind = tf.range(tf.shape(pick_boxes2)[0])
        box_ind = tf.zeros((tf.shape(pick_boxes2)[0]), dtype=tf.int32)
        shp_int = tf.dtypes.cast(img_shp, tf.int32)
        image = tf.reshape(image_data, (1,shp_int[0], shp_int[1], shp_int[2]))
        crop_imgs = tf.image.crop_and_resize(image, crop_boxes, box_ind=box_ind, crop_size=(24,24))
        crop_imgs = (crop_imgs-127.5)*0.0078125
        crop_imgs = tf.transpose(crop_imgs, (0,2,1,3)) # where suspect
        crop_imgs = tf.reshape(crop_imgs, (-1, 24, 24, 3), name='input')
        map02 = (-1)*tf.ones((1, 24, 24, 3))
        box02 = tf.zeros((1, 5))
        pick_boxes2 = tf.concat([pick_boxes2, box02], axis=0) # y, x
        crop_imgs = tf.concat([crop_imgs, map02], axis=0)
        rnet = RNet({'data':crop_imgs})
        rout = [rnet.layers['conv5-2'], rnet.layers['prob1']]
        out0 = rout[0]
        out1 = tf.transpose(rout[1])
        score = out1[1, :]
        ipass = tf.where(score>0.7)
        pick_boxes2 = tf.gather_nd(pick_boxes2[:, :4], ipass)
        score = tf.gather_nd(score, ipass)
        score = tf.transpose(tf.expand_dims(score, 0))
        pick_boxes2 = tf.concat([pick_boxes2, score], axis=1)
        mv = tf.gather_nd(out0, ipass)
        nms_inds2 = tf.image.non_max_suppression(pick_boxes2[:,:4], pick_boxes2[:, 4], 1000, 0.7)
        pick_boxes2 = tf.gather(pick_boxes2, nms_inds2)
        mv = tf.gather(mv, nms_inds2)
        pick_boxes2 = bbreg(pick_boxes2, mv)
        pick_boxes2 = rerec(pick_boxes2)
        pick_boxes2 = simple_crop(pick_boxes2, w_img, h_img)
        load_np(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet'):
        pick_boxes3 = pick_boxes2
        box_ind = tf.zeros((tf.shape(pick_boxes3)[0]), dtype=tf.int32)
        crop_boxes2 = transform_fpcoor_for_tf(pick_boxes3[:, :4], img_shp, [48, 48])
        crop_imgs2 = tf.image.crop_and_resize(image, crop_boxes2, box_ind=box_ind, crop_size=(48,48))
        crop_imgs2 = (crop_imgs2-127.5)*0.0078125
        crop_imgs2 = tf.transpose(crop_imgs2, (0,2,1,3)) # where suspect
        crop_imgs2 = tf.reshape(crop_imgs2, (-1, 48, 48, 3), name='input')
        map03 = (-1)*tf.ones((1, 48, 48, 3))
        box03 = tf.zeros((1, 5))
        pick_boxes3 = tf.concat([pick_boxes3, box03], axis=0)
        crop_imgs2 = tf.concat([crop_imgs2, map03], axis=0)
        onet = ONet({'data':crop_imgs2})
        out = [onet.layers['conv6-2'],onet.layers['conv6-3'],onet.layers['prob1']]
        out0 = out[0]
        out1 = out[1]
        out2 = out[2]
        score = out2[:, 1]
        ipass = tf.where(score>0.7)
        points = tf.gather_nd(out1, ipass)
        mv = tf.gather_nd(out0, ipass)
        pick_boxes3 = tf.gather_nd(pick_boxes3, ipass)
        score = tf.reshape(tf.gather_nd(score, ipass), [-1, 1])
        pick_boxes3 = tf.concat([pick_boxes3[:, :4], score], axis=1)
        w = pick_boxes3[:,2]-pick_boxes3[:,0]+1
        h = pick_boxes3[:,3]-pick_boxes3[:,1]+1
        w = tf.reshape(w, [-1, 1])
        h = tf.reshape(h, [-1, 1])
        x0 = tf.reshape(pick_boxes3[:, 0], [-1, 1])
        y0 = tf.reshape(pick_boxes3[:, 1], [-1, 1])
        ptX = points[:, :5]*w+x0-1
        ptY = points[:, 5:]*h+y0-1
        pointsXY = tf.concat([ptX, ptY], axis=1)
        pick_boxes3 = bbreg(pick_boxes3, mv)
        x1 = pick_boxes3[:, 0]
        y1 = pick_boxes3[:, 1]
        x2 = pick_boxes3[:, 2]
        y2 = pick_boxes3[:, 3]
        s = pick_boxes3[:, 4]
        nms_boxes3 = tf.transpose(tf.stack([y1, x1, y2, x2], axis=0))
        nms_inds3 = tf.image.non_max_suppression(nms_boxes3, s, 500, iou_threshold=0.5)
        boxes = tf.gather(pick_boxes3, nms_inds3, name='boxes')
        pointsXY = tf.gather(pointsXY, nms_inds3, name='points')
        load_np(os.path.join(model_path, 'det3.npy'), sess)
        box_ind = get_minIou_tf(boxes)
        boxes = tf.gather(boxes, box_ind)
        pointsXY = tf.gather(pointsXY, box_ind)

    return [boxes, pointsXY], image_data


def get_minIou_tf(boxes):
    box_len = tf.shape(boxes)[0]
    x1 = tf.reshape(boxes[:, 0], [1, -1])
    y1 = tf.reshape(boxes[:, 1], [1, -1])
    x2 = tf.reshape(boxes[:, 2], [1, -1])
    y2 = tf.reshape(boxes[:, 3], [1, -1])
    x1t = tf.reshape(boxes[:, 0], [-1, 1])
    y1t = tf.reshape(boxes[:, 1], [-1, 1])
    x2t = tf.reshape(boxes[:, 2], [-1, 1])
    y2t = tf.reshape(boxes[:, 3], [-1, 1])
    x1 = tf.tile(x1, [box_len, 1])
    x1t = tf.tile(x1t, [1, box_len])
    x1Mat = tf.where(x1>x1t, x1, x1t)
    y1 = tf.tile(y1, [box_len, 1])
    y1t = tf.tile(y1t, [1, box_len])
    y1Mat = tf.where(y1>y1t, y1, y1t)
    x2 = tf.tile(x2, [box_len, 1])
    x2t = tf.tile(x2t, [1, box_len])
    x2Mat = tf.where(x2<x2t, x2, x2t)
    y2 = tf.tile(y2, [box_len, 1])
    y2t = tf.tile(y2t, [1, box_len])
    y2Mat = tf.where(y2<y2t, y2, y2t)
    xMat = x2Mat-x1Mat
    xMat = tf.where(xMat>0, xMat, tf.zeros(tf.shape(xMat)))
    yMat = y2Mat-y1Mat
    yMat = tf.where(yMat>0, yMat, tf.zeros(tf.shape(yMat)))
    iouMat = xMat*yMat
    eye = 1-tf.eye(box_len)
    iouMat = iouMat*eye
    maxIou = tf.math.reduce_max(iouMat, 1)
    area = ((x2-x1)*(y2-y1))[0, :]
    minRate = maxIou/area
    box_ind = tf.where(minRate<0.7)
    box_ind = tf.reshape(box_ind, [-1])
    return box_ind


def get_scales(img):
    minsize=20
    factor=0.709
    factor_count=0
    h=img.shape[0]
    w=img.shape[1]
    minl=np.amin([h, w])
    m=12.0/minsize
    minl=minl*m
    scales=[]
    while minl>=12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl*factor
        factor_count += 1
    print(scales)
    return scales

def build():
    '''
    input_node: pnet/input, pnet/scales, pnet/scale_len
    output_node: onet/boxes, onet/points
    '''
    imgPath = "C:\\Users\\admin\\Pictures\\stars.jpg"
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scales = get_scales(img)
    scales = np.array(scales, dtype=np.float32)
    sess = tf.Session()
    result, image_data1 = MyMtcnnNet(sess)
    boxes, points = sess.run(result, feed_dict={image_data1:img})
    # print(boxes, boxes.shape)
    # g = tf.get_default_graph().as_graph_def()
    # for n in g.node:
    #     print(n.name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    drawFaces(img, list(boxes), list(points))

def drawFaces(img, boxes, points):
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
        if score < 0.80:
            continue
        mid = (int((b0 + b2) / 2) - 15, int((b1 + b3) / 2)+5)
        top = (int((b0 + b2) / 2) - 15, b1-4)
        cv2.rectangle(img, (b0, b1), (b2, b3), (0,0,255), 2)
        cv2.putText(img, str(score*100)[:4], top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        pts = points[i].astype(np.int32)
        ptsX = pts[:5]
        ptsY = pts[5:]
        for i in range(ptsX.shape[0]):
            cv2.circle(img, (ptsX[i], ptsY[i]), 2, (0,0,255))
    # img = cv2.resize(img, (int(img.shape[1]*0.6), int(img.shape[0]*0.6)), interpolation=cv2.INTER_AREA)
    cv2.imshow('faces', img)
    return cv2.waitKey(0)


if __name__ == "__main__":
    build()