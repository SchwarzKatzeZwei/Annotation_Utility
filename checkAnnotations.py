# -*- coding: utf-8 -*-
import argparse
import glob
import os
import xml.etree.ElementTree as ET

def getParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xmldir",
                        help="XML(Annotaions) directory path",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--outdir",
                        help="crop image output directory path",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def question_input(question):
    while True:
        choice = input(question).lower()
        if choice in ['d', 'delete']:
            return "delete"
        elif choice in ['c', 'continue']:
            return "continue"
        elif choice in ["s", "stop"]:
            return "stop"
        else:
            print("Answer Error!")
            exit(1)

def main(args):
    fileList = sorted(glob.glob(args.xmldir + "/*.xml"))
    for file in fileList:
        print(file)
        tree = ET.parse(file)
        root = tree.getroot()

        for obj in root.findall(".//object"):
            _xmin = int(obj.find(".//xmin").text)
            _ymin = int(obj.find(".//ymin").text)
            _xmax = int(obj.find(".//xmax").text)
            _ymax = int(obj.find(".//ymax").text)
            if _xmax - _xmin < 20 and _ymax - _ymin:
                print(f"  ({_xmin}, {_ymin}), ({_xmax}, {_ymax})size:({_xmax - _xmin}:{_ymax - _ymin})")
                ret = question_input("  delete(d),continue(c),stop(s)")
                if ret == "delete":
                    root.remove(obj)
                elif ret == "continue":
                    pass
                elif ret == "stop":
                    exit(0)

        tree.write(args.outdir + "/" + os.path.basename(file))


if __name__ == "__main__":
    _args = getParse()
    main(_args)
