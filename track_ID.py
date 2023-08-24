# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from timeit import time
import warnings
import argparse

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

from yolo_v3 import YOLO3
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID
import copy
#
# 
# --videos ./videos/init/Single1.mp4 

parser = argparse.ArgumentParser()
parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v3')
parser.add_argument('--videos', nargs='+', help='List of videos', required=False, default='./videos/init/4.avi')
parser.add_argument('--camera', default=0, help='List of camera')
parser.add_argument('-all', help='Combine all videos into one', default=True)
parser.add_argument('--save_result2video', required=False, default=False)
args = parser.parse_args()  # vars(parser.parse_args())


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh

class LoadVideo_from_camera:  # for inference
    def __init__(self, camera_ID, img_size=(1088, 608)):
        if not camera_ID:
            raise FileExistsError
        # cameraCapture = cv2.VideoCapture(camera_ID)
        self.cap = cv2.VideoCapture(camera_ID)
        self.success, self.frame = self.cap.read()
        print('============',np.array(self.frame).shape)
        self.vw = np.array(self.frame).shape[1]
        self.vh = np.array(self.frame).shape[0]
        print('W *H of camera {}: {:d} * {:d}'.format(camera_ID, self.vw, self.vh))

    def get_VideoLabels(self):
        return self.frame, self.success, self.vw, self.vh

class track_go():
    def __init__(self, yolo, target_video):
        print(f'Using {yolo} model')
        self.yolo = yolo
        self.is_vis = True
        self.max_cosine_distance = 0.2
        self.nn_budget = None
        self.nms_max_overlap = 0.4
        self.is_sub = False
        # deep_sort
        self.model_filename = 'model_data/models/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)  # use to get feature

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric, max_age=100)

        output_frames = []
        output_rectanger = []
        output_areas = []
        output_wh_ratio = []

        self.out_dir = 'videos/output/'
        print('The output folder is', self.out_dir)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.tracking_path = self.out_dir + 'tracking_camera' + '.avi'
        # self.combined_path = self.out_dir + 'allVideos' + '.avi'
        self.tracking_path = self.out_dir + 'camera_0' + '.avi'

        self.target_all_frames = []
        loadvideo = LoadVideo(target_video)
        video_capture, self.frame_rate, w, h = loadvideo.get_VideoLabels()
        while True:
            ret, frame = video_capture.read()
            if ret is not True:
                video_capture.release()
                break
            self.target_all_frames.append(frame)

        

        # all_frames = video_frame
        # frame_nums = len(all_frames)
        
        # self.combined_path = self.out_dir + 'tracking_camera_REID' + '.avi'
        # if self.is_vis:
        #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #     # self.out = cv2.VideoWriter(self.tracking_path, fourcc, frame_rate, (w, h))
        #     self.out2 = cv2.VideoWriter(self.combined_path, fourcc, self.frame_rate, (w, h))
        #     # Combine all videos
        #     for frame_all_index in self.target_all_frames:
        #         self.out2.write(frame_all_index)
  
        #     self.out2.release()    #init + track_camera

        # Initialize tracking file
        self.filename = self.out_dir + '/tracking.txt'
        open(self.filename, 'w')
        
    def track_ID(self, Frame):
        frame = Frame
        fps = 0.0
        frame_cnt = 0
        t1 = time.time()

        track_cnt = dict()
        images_by_id = dict()
        ids_per_frame = []
        
        # for frame in all_frames:
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb，最后一维通道逆序  array 转成image  (480, 640, 3)
        # print('=====2',image.height, image.width) #(480, 640, 3)
        w = image.width
        h = image.height
        boxs = self.yolo.detect_image(image)  # n * [topleft_x, topleft_y, w, h]
        features = self.encoder(frame, boxs)  # n * 128
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # length = n
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, self.nms_max_overlap, scores)
        # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # length = len(indices)

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        tmp_ids = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                    ]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append([
                        frame_cnt,
                        int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                        area
                    ])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            cv2_addBox(
                track.track_id,
                frame,
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                line_thickness,
                text_thickness,
                text_scale
            )
            write_results(
                self.filename,
                'mot',
                frame_cnt + 1,
                str(track.track_id),
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                w, h
            )
            # print('===****======write camera frame NO REID')
            # self.out = cv2.VideoWriter(self.tracking_path, fourcc, fps, size)
            

        print('=========write txt')
        ids_per_frame.append(set(tmp_ids))

        # save a frame
        # if self.is_vis:
        #     self.out.write(frame)
        #     print('=========write camera frame no REID',ids_per_frame)

        t2 = time.time()

        # frame_cnt += 1
        # print(frame_cnt, '/', frame_nums)

        print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
        # print('Tracked video : {}'.format(self.tracking_path))
        # print('Combined video : {}'.format(self.combined_path))

        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        reid = REID()
        threshold = 320
        exist_ids = set()
        final_fuse_id = dict()
        print('===============================REID, Total IDs =',len(images_by_id))
        # print(f'Total IDs = {len(images_by_id)}')
        feats = dict()
        for i in images_by_id:
            print(f'==========ID number {i} -> Number of frames {len(images_by_id[i])}')
            feats[i] = reid._features(images_by_id[i])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])

        for f in ids_per_frame:
            if f:
                if len(exist_ids) == 0:
                    for i in f:
                        final_fuse_id[i] = [i]
                    exist_ids = exist_ids or f  #并集
                else:
                    new_ids = f - exist_ids
                    for nid in new_ids:
                        dis = []
                        if len(images_by_id[nid]) < 10:
                            exist_ids.add(nid)
                            continue
                        unpickable = []
                        for i in f:
                            for key, item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        print('=============exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                        for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                            tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                            print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                            dis.append([oid, tmp])
                        exist_ids.add(nid)
                        if not dis:
                            final_fuse_id[nid] = [nid]
                            continue
                        dis.sort(key=operator.itemgetter(1))
                        if dis[0][1] < threshold:
                            combined_id = dis[0][0]
                            images_by_id[combined_id] += images_by_id[nid]
                            final_fuse_id[combined_id].append(nid)
                        else:
                            final_fuse_id[nid] = [nid]
        print('==============Final ids and their sub-ids:', final_fuse_id)
        print('MOT took {} seconds'.format(int(time.time() - t1)))
        t2 = time.time()

        # To generate MOT for each person, declare 'is_vis' to True
        # self.is_sub = False
        if self.is_sub:
            print('Writing videos for each ID...')
            output_dir = 'videos/output/tracklets/'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            loadvideo = LoadVideo(self.tracking_path)
            video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            for idx in final_fuse_id:
                tracking_path = os.path.join(output_dir, str(idx)+'.avi')
                out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
                for i in final_fuse_id[idx]:
                    for f in track_cnt[i]:
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                        # _, frame = video_capture.read()
                        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                        cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                        out.write(frame)
                        # print('===****======write camera frame YES REID')
                out.release()
            video_capture.release()


        # Generate a single video with complete MOT/ReID
        if args.all:  
            self.target_all_frames.append(frame)
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # complete_path = self.out_dir+'/Complete'+'.avi'
            # out = cv2.VideoWriter(complete_path, fourcc, self.frame_rate/10, (w, h))

            for frame_index in range(len(np.array(self.target_all_frames))):
                frame2 = self.target_all_frames[frame_index]
                # video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                # _, frame2 = video_capture.read()

                for idx in final_fuse_id:
                    for i in final_fuse_id[idx]:
                        for f in track_cnt[i]:
                            # print('frame2 {} f0 {}'.format(frame2,f[0]))
                            if frame_index == f[0]:
                                text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                                cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                
                # out.write(frame2)    
            # out.release()
            # video_capture.release()
        
        # os.remove(combined_path)
        # print('\nWriting videos took {} seconds'.format(int(time.time() - t2)))
        # print('Final video at {}'.format(complete_path))
        # print('Total: {} seconds'.format(int(time.time() - t1)))
        return frame, frame2
        # self.out.release()

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def ID_get(camera_ID_VideoPath):
    cameraCapture = cv2.VideoCapture(camera_ID_VideoPath)
    success, camera_frames = cameraCapture.read()
    fps = cameraCapture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('============H * W',np.array(camera_frames).shape)  #(480, 640, 3)
    w = np.array(camera_frames).shape[1]
    h = np.array(camera_frames).shape[0]
    track_camera = track_go(yolo=YOLO3() if args.version == 'v3' else YOLO4(), target_video=args.videos)
    # return track_camera

    track_camera_out = cv2.VideoWriter(track_camera.out_dir + 'tracking_camera0' + '.avi', fourcc, fps, size)
    track_camera_out_REID = cv2.VideoWriter(track_camera.out_dir + 'tracking_camera0_REID' + '.avi', fourcc, fps, size)
    while success:
        if cv2.waitKey(1) == 27:
            break
        print('==========Tracking and REID Processing....')
        frame_result, frame_result_REID= track_camera.track_ID(camera_frames)
        # if args.save_result2video: 
        track_camera_out.write(frame_result)
        cv2.waitKey(1)
        Is_REID = True
        if Is_REID:
            track_camera_out_REID.write(frame_result_REID)
            cv2.waitKey(1)
        success, camera_frames = cameraCapture.read()
        # cv2.namedWindow('Final_tracking_REID')
        # cv2.imshow('Final_tracking_REID',frame_result)
        # cv2.waitKey(1)



# ID_get(0)
