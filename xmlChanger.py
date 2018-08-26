# -*- coding: utf-8 -*-
import argparse
from xml.etree import ElementTree
import os
import glob
import cv2
import numpy as np

def getParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir",
                        help="source annotations directory",
                        required=True,
                        type=str)

    _args = parser.parse_args()
    return _args

def hsvAllAverage(hsv_image):
    hue_ave = np.mean(hsv_image[:, :, 0])
    sat_ave = np.mean(hsv_image[:, :, 1])
    bri_ave = np.mean(hsv_image[:, :, 2])

    return [hue_ave, sat_ave, bri_ave]

def horizonConcat(img):
    """画像横連結
    img : [img1, img2, ...]"""
    return cv2.hconcat(img)

def closeStop():
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

def main(args):
    xml_list = sorted(glob.glob(os.path.join(args.dir, '*.xml')))
    for i, xml in enumerate(xml_list):
        print('{}/{} {}'.format(i + 1, len(xml_list), xml))

        # 画像読込
        img_path = args.dir + "../train_images/" + os.path.splitext(os.path.basename(xml))[0] + ".jpg"
        img_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        img_rgb2 = cv2.imread(args.dir + "../kukei_images/" + os.path.splitext(os.path.basename(xml))[0] + ".jpg")

        # XML Parse
        tree = ElementTree.parse(xml)
        root = tree.getroot()
        for obj in root.findall('object'):
            position = []

            for bb in obj.find("bndbox"):
                position.append(int(bb.text))
            ave = hsvAllAverage(img_hsv[position[1]:position[3], position[0]:position[2]])
            if ave[0] < 89:
                color = (128, 128, 128)
                nametag = "warm"
            elif ave[0] < 130:
                color = (23, 232, 166)
                nametag = "cool"
            else:
                color = (0, 102, 7)
                nametag = "cold"

            font = cv2.FONT_HERSHEY_SIMPLEX
            try:
                cv2.putText(img_rgb2, str(int(ave[0])), (position[0] + int((position[2] - position[0]) / 3),
                                                         position[1] + int((position[3] - position[1]) / 1.5)), font, 0.6, color, 2, 3)
            except ValueError:
                print(position)

            obj.find("name").text = nametag
        tree.write("new_xml/"+os.path.basename(xml))

        #cv2.imshow("win", img_rgb2)
        #cv2.moveWindow("win", 1921, 0)
        #closeStop()

if __name__ == "__main__":
    _args = getParse()
    main(_args)
