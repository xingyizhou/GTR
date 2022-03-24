import argparse
import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path

NC = 1300
COLORS = [((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8) \
              for _ in range(NC)]

# SKIP_IDS = ['02', '04', '05', '09', '10', '11', '13']
SKIP_IDS = []

def draw_bbox(img, bboxes, K, traces):
    for bbox in bboxes:
        track_id = int(bbox[4])
        # c = COLORS[int(bbox[4]) % NC]
        c = np.array(COLORS[track_id % NC]).astype(np.int32).tolist()
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
            c, 2, lineType=cv2.LINE_AA)
        ct = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        txt = '{}'.format(track_id)
        cv2.putText(img, txt, (int(ct[0]), int(ct[1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                c, thickness=1, lineType=cv2.LINE_AA)
        if traces is not None:
            traces[K][track_id].append(
                (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3]))
            for p in traces[K][track_id]:
                cv2.circle(
                    img, (int(p[0]), int(p[1])), 2, c, -1, 
                    lineType=cv2.LINE_AA)
    return traces

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', default='datasets/mot/MOT17/test/')
    parser.add_argument('--preds')
    parser.add_argument('--resize', type=int, default=2)
    parser.add_argument('--save_path', default='')
    parser.add_argument('--is_gt', action='store_true')
    parser.add_argument('--show_trace', action='store_true')
    args = parser.parse_args()
    seqs = os.listdir(args.gt_path)
    save_video = args.save_path != ''
    if save_video:
        save_path = args.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print('save_video_path', save_path)
    vised = set()
    for seq in sorted(seqs):
        print('seq', seq)
        seq_name = seq[:seq.rfind('-')]
        for skip in SKIP_IDS:
            if skip in seq_name:
                continue
        if seq_name in vised:
            continue
        vised.add(seq_name)
        if '.DS_Store' in seq:
            continue
        seq_path = '{}/{}/'.format(args.gt_path, seq)
        # if args.is_gt:
        #     ann_path = seq_path + 'gt/gt.txt'
        # else:
        #     ann_path = seq_path + 'det/det.txt'
        # anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        # print('anns shape', anns.shape)
        # image_to_anns = defaultdict(list)
        # for i in range(anns.shape[0]):
        #     if (not args.is_gt) or (int(anns[i][6]) == 1 and float(anns[i][8]) >= 0.25):
        #         frame_id = int(anns[i][0])
        #         track_id = int(anns[i][1])
        #         bbox = (anns[i][2:6] / args.resize).tolist()
        #         image_to_anns[frame_id].append(bbox + [track_id])
        
        image_to_preds = {}
        preds_name = args.preds.split(',')
        for K, pred_name in enumerate(preds_name):
            num_boxes = 0
            tracks = set()
            image_to_preds[K] = defaultdict(list)
            pred_path = pred_name + '/{}.txt'.format(seq)
            try:
                preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=',')
            except:
                preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=' ')
            for i in range(preds.shape[0]):
                frame_id = int(preds[i][0])
                track_id = int(preds[i][1])
                bbox = (preds[i][2:6] / args.resize).tolist()
                image_to_preds[K][frame_id].append(bbox + [track_id])
                num_boxes += 1
                tracks.add(track_id)
            print('num_boxes, num_tracks', K, num_boxes, len(tracks))
        # continue
        img_path = seq_path + 'img1/'
        images = os.listdir(img_path)
        num_images = len([image for image in images if 'jpg' in image])
        if args.show_trace:
            traces = [defaultdict(list) for _ in preds_name]
        else:
            traces = None
        for i in range(num_images):
            frame_id = i + 1
            file_name = '{}/img1/{:06d}.jpg'.format(seq, i + 1)
            file_path = args.gt_path + '/' + file_name
            img = cv2.imread(file_path)
            if args.resize != 1:
                img = cv2.resize(
                    img, 
                    (img.shape[1] // args.resize, img.shape[0] // args.resize))
            if save_video and i == 0:
                H, W = img.shape[:2]
                videos = []
                for K, _ in enumerate(preds_name):
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    videos.append(cv2.VideoWriter(
                        '{}/{}_{}.mkv'.format(save_path, seq, K), 
                        fourcc, 20.0, (W, H), isColor=True)
                    )
                print(W, H)
            for K, pred_name in enumerate(preds_name):
                img_pred = img.copy()
                traces = draw_bbox(
                    img_pred, image_to_preds[K][frame_id], K, traces)
                if save_video:
                    videos[K].write(img_pred)
                else:
                    cv2.imshow('pred{}'.format(K), img_pred)
            if not save_video:
                cv2.waitKey()
        if save_video:
            for K, _ in enumerate(preds_name):
                videos[K].release()