# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ["person", "car"]  # 与 LabelImg / predefined_classes.txt 一致
abs_path = os.getcwd()
print(abs_path)


def convert(size, box):
    # box: xmin, xmax, ymin, ymax -> YOLO cx, cy, w, h (normalized)
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_annotation(image_id):
    in_file = open('xml/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open('labels/%s.txt' % (image_id), 'w', encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult_node = obj.find('difficult')
        difficult = 0 if difficult_node is None else int(difficult_node.text)
        cls = obj.find('name').text
        if cls not in classes or difficult == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


wd = getcwd()
for image_set in sets:
    if not os.path.exists('labels/'):
        os.makedirs('labels/')
    image_ids = open('dataSet/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        linestr=os.path.join(abs_path,"images","%s.jpg"%(image_id))
        list_file.write(linestr+"\n")
        print(image_id)
        convert_annotation(image_id)
    list_file.close()