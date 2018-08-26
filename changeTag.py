# -*- coding: utf-8 -*-
"""
    Annotationファイルからタグ名を強制変更
"""
import os
import argparse
from xml.dom import minidom
from tqdm import tqdm

def getParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--srcxmldir",
                        help="source XML(Annotaions) directory path",
                        required=True,
                        type=str)
    parser.add_argument("-d", "--dstxmldir",
                        help="destination XML(Annotations) directory path",
                        required=True,
                        type=str)
    parser.add_argument("-t", "--tag",
                        help="change tag",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def main(args):
    for file in tqdm(sorted(os.listdir(args.srcxmldir))):
        filepath = args.srcxmldir + file
        xml = minidom.parse(filepath)
        for i in range(0, len(xml.getElementsByTagName("name"))):
            xml.getElementsByTagName("name")[i].childNodes[0].data = args.tag
        with open(args.dstxmldir + file, mode="w") as f:
            f.write(xml.toxml())

if __name__ == "__main__":
    _args = getParse()
    main(_args)
