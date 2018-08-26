# -*- coding: utf-8 -*-
import argparse
import os
import glob
import sys
sys.path.append("../")
sys.path.append("../ssd_util/")
from xml.etree import ElementTree
from xml.dom import minidom
from collections import namedtuple
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

# GPU Memory Saving Mode ------------------------
from keras.backend import tensorflow_backend
import tensorflow as tf
CONFIG = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",
                                                  allow_growth=True))
SESSION = tf.Session(config=CONFIG)
tensorflow_backend.set_session(SESSION)
# -----------------------------------------------

from ssd_util.ssd import SSD300
from ssd_util.ssd_utils import BBoxUtility

IMAGE_SIZE = namedtuple('IMAGE_SIZE', ('width', 'height', 'depth'))
THRESHOLD_CLASS = {"ALL"  :0.80,
                   "panel": 0.80}

def getParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        help="model path",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir",
                        help="output annotation directory",
                        required=False,
                        default="results",
                        type=str)
    parser.add_argument("--image_dir",
                        help="image directory",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def load_bbox_label_names():
    if not os.path.isfile('labels.txt'):
        print('labels.txt is not exist.')
        exit()

    labels = []
    with open('labels.txt') as fp:
        for label in fp:
            labels.append(label.replace('\n', ''))

    return tuple(labels)

def create_pascalVOC(full_name, img_size, labels, bbox, output_file_name):
    def prettify(elem):
        rough_string = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    top = ElementTree.Element('annotation')

    dir_name, file_name = os.path.split(full_name)
    folder = ElementTree.SubElement(top, 'folder')
    folder.text = str(dir_name)

    filename = ElementTree.SubElement(top, 'filename')
    filename.text = str(file_name)

    path = ElementTree.SubElement(top, 'path')
    path.text = str(full_name)

    source = ElementTree.SubElement(top, 'source')
    source.text = 'Unknown'
    # owner = ElementTree.SubElement(top, 'owner')

    size_s = ElementTree.SubElement(top, 'size')
    w = ElementTree.SubElement(size_s, 'width')
    w.text = str(img_size.width)
    h = ElementTree.SubElement(size_s, 'height')
    h.text = str(img_size.height)
    d = ElementTree.SubElement(size_s, 'depth')
    d.text = str(img_size.depth)

    seg = ElementTree.SubElement(top, 'segmented')
    seg.text = str(0)

    for c in range(len(labels)):
        _object = ElementTree.SubElement(top, 'object')

        name = ElementTree.SubElement(_object, 'name')
        name.text = str(labels[c])

        pose = ElementTree.SubElement(_object, 'pose')
        pose.text = 'Unspecified'

        truncated = ElementTree.SubElement(_object, 'truncated')
        truncated.text = str(1)

        difficult = ElementTree.SubElement(_object, 'difficult')
        difficult.text = str(0)

        bboxElm = ElementTree.SubElement(_object, 'bndbox')
        xmin = ElementTree.SubElement(bboxElm, 'xmin')
        xmin.text = str(int(bbox[c][1]))
        ymin = ElementTree.SubElement(bboxElm, 'ymin')
        ymin.text = str(int(bbox[c][0]))
        xmax = ElementTree.SubElement(bboxElm, 'xmax')
        xmax.text = str(int(bbox[c][3]))
        ymax = ElementTree.SubElement(bboxElm, 'ymax')
        ymax.text = str(int(bbox[c][2]))

    elm = prettify(top)
    with open(output_file_name, 'w') as fp:
        fp.write(elm)

def validImageSet(imagepath):
    """検証画像をメモリにセットする関数"""
    def kerasLoadImage(filepath, target_size=(300, 300)):
        img = image.load_img(filepath, target_size=target_size)
        img = image.img_to_array(img)
        return img

    valid_imgs = []
    inputs = []
    valid_imgs.append([os.path.basename(imagepath), cv2.imread(imagepath)])
    inputs.append(kerasLoadImage(imagepath))
    np_inputs = preprocess_input(np.array(inputs))

    return valid_imgs, np_inputs

def predict(np_input, valid_imgs, model, bbox_util, classes):
    labels = []
    bboxes = []
    preds = model.predict(np_input, batch_size=4, verbose=0)
    results = bbox_util.detection_out(preds)
    for i, img in enumerate(valid_imgs):
        img = img[1]
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than PREDICT_RATE.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= THRESHOLD_CLASS["ALL"]]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for k in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[k] * img.shape[1]))
            ymin = int(round(top_ymin[k] * img.shape[0]))
            xmax = int(round(top_xmax[k] * img.shape[1]))
            ymax = int(round(top_ymax[k] * img.shape[0]))
            label = int(top_label_indices[k])
            label_name = classes[label - 1]

            labels.append(label_name)
            bboxes.append([ymin, xmin, ymax, xmax])

    return labels, bboxes

def main():
    args = getParse()

    # 出力ディレクトリ準備
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print('output dir is exist.')
        exit()

    # ラベルロード
    bbox_label_names = load_bbox_label_names()
    print(bbox_label_names)

    model = SSD300((300, 300, 3), num_classes=len(bbox_label_names)+1)
    model.load_weights(args.model, by_name=True)
    bbox_util = BBoxUtility(len(bbox_label_names)+1)

    jpg_list = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))

    for i, jpg_name in enumerate(jpg_list):
        print('{}/{} {}'.format(i + 1, len(jpg_list), jpg_name))

        # object detection
        valid_imgs, np_inputs = validImageSet(jpg_name)
        labels, bboxes = predict(np_inputs, valid_imgs, model, bbox_util, bbox_label_names)

        # create_output_dir and name
        filename = os.path.split(jpg_name)[1]
        output_name = os.path.join(args.output_dir, (os.path.splitext(filename)[0] + '.xml'))

        full_name = os.path.abspath(jpg_name)
        img_size = IMAGE_SIZE(valid_imgs[0][1].shape[1], valid_imgs[0][1].shape[0], valid_imgs[0][1].shape[2])
        create_pascalVOC(full_name, img_size, labels, bboxes, output_name)

        #print(f"{full_name}:{img_size}:{output_name}")


if __name__ == "__main__":
    main()
