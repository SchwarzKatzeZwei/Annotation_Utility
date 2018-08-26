# -*- coding: utf-8 -*-
"""
    Annotationファイルから画像をクロップ
"""
import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

def getParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xmldir",
                        help="XML(Annotaions) directory path",
                        required=True,
                        type=str)
    parser.add_argument("-i", "--imgdir",
                        help="image directory path",
                        required=False,
                        default="results",
                        type=str)
    parser.add_argument("-o", "--outdir",
                        help="crop image output directory path",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def imgcropout(imgfile, name, xmin, ymin, xmax, ymax, counter, outFilePath):
    im = Image.open(imgfile)
    im_crop = im.crop((xmin, ymin, xmax, ymax))
    outfilename = outFilePath + name + "_" + str(counter).zfill(5) + ".jpg"
    im_crop.save(outfilename, quality=100)
    return counter + 1

def main(args):
    counter = 1
    for file in tqdm(sorted(os.listdir(args.xmldir))):
        filename = args.xmldir + file
        tree = ET.parse(filename)
        root = tree.getroot()

        for obj in root.findall(".//object"):
            _name = obj.find(".//name").text
            _xmin = int(obj.find(".//xmin").text)
            _ymin = int(obj.find(".//ymin").text)
            _xmax = int(obj.find(".//xmax").text)
            _ymax = int(obj.find(".//ymax").text)

            # 画像出力
            counter = imgcropout(args.imgdir + root.find("filename").text, _name, _xmin, _ymin, _xmax, _ymax, counter, args.outdir)

if __name__ == "__main__":
    _args = getParse()
    main(_args)
