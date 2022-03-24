
import os
import numpy as np
import json
import cv2

DATA_PATH = 'datasets/crowdhuman/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['val', 'train']
IMAGE_DIR = 'datasets/crowdhuman/CrowdHuman_{}/Images/'

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

if __name__ == '__main__':
  if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)
  for split in SPLITS:
    data_path = DATA_PATH + split
    out_path = OUT_PATH + '{}_amodal.json'.format(split)
    out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'person'}]}
    ann_path = DATA_PATH + '/annotation_{}.odgt'.format(split)
    anns_data = load_func(ann_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for i, ann_data in enumerate(anns_data):
      if i % 1000 == 0:
        print(i, 'of', len(anns_data))
      image_cnt += 1

      file_name = '{}.jpg'.format(ann_data['ID'])
      image_path = IMAGE_DIR.format(split) + file_name
      img = cv2.imread(image_path)
      h, w = img.shape[:2]
      image_info = {
          'file_name': file_name,
          'id': image_cnt,
          'height': h,
          'width': w,
        }
      out['images'].append(image_info)

      if split != 'test':
        anns = ann_data['gtboxes']
        for i in range(len(anns)):
          ann_cnt += 1
          ann = {'id': ann_cnt,
                 'category_id': 1,
                 'image_id': image_cnt,
                 'bbox_vis': anns[i]['vbox'],
                 'bbox': anns[i]['fbox'],
                 'iscrowd': 1 if 'extra' in anns[i] and \
                                 'ignore' in anns[i]['extra'] and \
                                 anns[i]['extra']['ignore'] == 1 else 0}
          ann['area'] = ann['bbox'][2] * ann['bbox'][3]
          out['annotations'].append(ann)
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
        
