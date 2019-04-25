# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:55:34 2019

@author: 50568
"""

import data_file_helper as fh
import tensorflow as tf
# import tensorflow.contrib as tf_contrib
import numpy as np
import cv2
import visualization


class data_image_helper:
    def __init__(self, detector):
        self.detector = detector

    def read_img(self, path, shape, size, begin=0, end=0):
        """
            Video_Read is used to extract the image of mouth from a video;\n
            parameter:\n
            Path: the string path of video\n
            Shape: the (min, max) size tuple of the mouth you extract from the video\n
            Size: the (high, weight) size tuple of the mouth image you save
        """
        cap = cv2.VideoCapture(path)
        images = []
        mouth = None
        cnt = 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(frames)
        print(fps)
        v_length = frames / fps
        if (end == 0 or end >= v_length):
            end = v_length

        if (cap.isOpened() == False):
            print("Read video failed!")
            return None

        # get detector
        classifier_face = cv2.CascadeClassifier(
            "./cascades/haarcascade_frontalface_alt.xml")
        classifier_mouth = cv2.CascadeClassifier(
            "./cascades/haarcascade_mcs_mouth.xml")

        cap.set(cv2.CAP_PROP_POS_FRAMES, begin * fps)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        while (pos <= end * fps and end <= v_length):
            ret, img = cap.read()
            '''
                第一个参数ret的值为True或False，代表有没有读到图片
                第二个参数是frame，是当前截取一帧的图片
            '''
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if ret == False:
                break

            faceRects_face = classifier_face.detectMultiScale(
                img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 25))
            key = cv2.waitKey(1)
            # 键盘等待
            if len(faceRects_face) > 0:
                # 检测到人脸
                for faceRect_face in faceRects_face:
                    # 获取图像x起点,y起点,宽，高
                    x, y, w, h = faceRect_face
                    # 转换类型为int，方便之后图像截取
                    intx = int(x)
                    intw = int(w)
                    # 截取人脸区域下半部分左上角的y起点，以精确识别嘴巴的位置
                    my = int(float(y + 0.6 * h))
                    mh = int(0.5 * h)

                    img_facehalf_bottom = img[my:(my + mh), intx:intx + intw]
                    '''
                        img获取坐标为，【y,y+h之间（竖）：x,x+w之间(横)范围内的数组】
                        img_facehalf是截取人脸识别到区域上半部分
                        img_facehalf_bottom是截取人脸识别到区域下半部分
                    '''
                    cv2.rectangle(img, (int(x), my),
                                  (int(x) + int(w), my + mh), (0, 255, 0), 2,
                                  0)
                    '''
                        矩形画出区域 rectangle参数（图像，左顶点坐标(x,y)，右下顶点坐标（x+w,y+h），线条颜色，线条粗细）
                        画出人脸识别下部分区域，方便定位
                    '''
                    faceRects_mouth = classifier_mouth.detectMultiScale(
                        img_facehalf_bottom, 1.1, 1, cv2.CASCADE_SCALE_IMAGE,
                        shape)
                    if len(faceRects_mouth) > 0:
                        for faceRect_mouth in faceRects_mouth:
                            xm1, ym1, wm1, hm2 = faceRect_mouth
                            cv2.rectangle(
                                img_facehalf_bottom, (int(xm1), int(ym1)),
                                (int(xm1) + int(wm1), int(ym1) + int(hm2)),
                                (0, 0, 255), 2, 0)

                            mouth = img_facehalf_bottom[ym1:(ym1 + hm2), xm1:(
                                xm1 + wm1)]
                            mouth = cv2.resize(
                                mouth, size, interpolation=cv2.INTER_CUBIC)

                            images.append(mouth)
                            cnt += 1
                            # cv2.imshow('video', mouth)
                            # if cnt % 10 == 0:
                            #     cv2.imwrite(str(cnt) + 'xx.jpg', mouth)

            if (key == ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()

        return images, cnt

    # def prepare_data(self,
    #                  path,
    #                  batch_size,
    #                  time_step,
    #                  shape = (20, 20),
    #                  size = (109, 109),
    #                  read = True):
    #     if(read):
    #         self.read_img(path, shape, size)

    #     DataSet = []
    #     Buffers = [None] * time_step
    #     cnt = 0

    #     for image in self.images:
    #         cnt += 1
    #         for i in range(time_step):
    #             Buffers[time_step -i - 1] = Buffers[time_step - i - 2]
    #         Buffers[0] = image

    #         if(cnt >= time_step):
    #             DataSet.append(Buffers.copy())

    #     # DataSet = DataSet / 255.0
    #     # DataSet = DataSet.astype(np.float32)

    #     batch_dataset = tf.data.Dataset.from_tensor_slices(DataSet)
    #     batch_dataset = batch_dataset.batch(batch_size)

    #     return batch_dataset, self.images

    def get_raw_dataset(self, path, shape=(20, 20), size=(224, 224)):

        video, cnt = self.read_img(path, shape, size, 0.5, 1)
        video = np.array(video) / 255.0
        video = video.astype(np.float32)
        return video
        # return tf.data.Dataset.from_generator(generator, tf.float32)

    def prepare_data(
            self,
            paths,
            batch_size,
            shape=(20, 20),
            size=(224, 224),
    ):

        dataset = []
        length = []
        for path in paths:
            video, cnt = self.read_img(path, shape, size, 0.5, 1)
            video = np.array(video) / 255.0
            video = video.astype(np.float32)
            dataset.append(video)
            length.append(cnt)

        def generator():
            for d, c in zip(dataset, length):
                yield d, c

        raw_dataset = tf.data.Dataset.from_generator(generator,
                                                     (tf.float32, tf.int32))
        batch_dataset = raw_dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, 109, 109, 5]),
                           tf.TensorShape([])))

        return batch_dataset, raw_dataset


if __name__ == '__main__':
    video, txt = fh.read_file(
        '/Users/barid/Documents/workspace/batch_data/lip_data')
    print(video[:5])
    print(txt[:5])
    helper = data_image_helper(detector='./cascades/')
    # b, d = helper.prepare_data(paths = ['D:/lip_data/ABOUT/train/ABOUT_00003.mp4'], batch_size = 64)
    b, d = helper.prepare_data(paths=video, batch_size=32)
    print(b)
#for (i,(x, l)) in enumerate(b):
#    print(x)
#    print(l)
