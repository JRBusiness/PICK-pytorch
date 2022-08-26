import json
import os
import shutil
from glob import glob
import ast
import csv
import cv2
from paddleocr import PaddleOCR
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import math
from sklearn.model_selection import train_test_split


class_index = {
    'card': 1,
    'dob': 2,
    'rxgroup': 3,
    'plan': 4,
    'health_plan': 5,
    'mem_name': 6,
    'payer_id': 7,
    'dependents': 8,
    'mem_id': 9,
    'effective': 10,
    'coverage': 11,
    'subcriber_id': 12,
    'pcp': 13,
    'service_type': 14,
    'provider_name': 15,
    'rxbin': 16,
    'group_number': 17,
    'rxpcn': 18,
    'issuer': 19,
}


def crop_img(image, polygon):
    top_left = tuple(int(val) for val in polygon[0])
    bottom_right = tuple(int(val) for val in polygon[2])
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]


def get_bbox(bbox):
    list_items = []
    for bb in bbox:
        # for i in ["x", "y"]:
        list_items.extend([bb["x"], bb["y"]])
    return list_items


def merge_boxes(box1, box2):
    return [
        box1[0], box2[0],
        box1[1], box2[1],
        box1[2], box2[2],
        box1[3], box2[3]
    ]


def processing_image(directory, line, class_writer, index):
    name = line['documentName'].split('.jpg_')[0]
    writer = csv.writer(open(f"data/{directory}/boxes_and_transcripts/{name}.tsv", "w", newline=""))
    final = []
    class_text = {}
    if line['annotation']:
        bboxs = line['annotation']
        if bboxs:
            inner_dex = 0
            for item in bboxs:
                inner_dex += 1
                bbox_line = {}
                text_line = {}
                label = item['label']
                text_line[label] = []
                bbox_line[label] = []
                seen = set()
                for box in item['boundingBoxes']:
                    bbox = box['normalizedVertices']
                    text = box['word']
                    text_line[label].append(text)
                    bbox = get_bbox(bbox)
                    if label in seen and bbox:
                        text_line[label] = [" ".join([i for i in text_line[label]])]
                        bbox_line[label] = [merge_boxes(bbox_line[label][0], bbox)]
                    elif label not in seen and bbox:
                        bbox_line[label].append(bbox)
                        seen.add(label)
                bbox_line[label][0].extend([text_line[label][0], label])
                class_text[label] = text_line[label][0]
                for k, v in bbox_line.items():
                    v[0].insert(0, inner_dex)
                    final.append(v[0])
    writer.writerows(final)
    json.dump(class_text, open(f"data/{directory}/entities/{name}.txt", "w", newline=""), indent=4)
    class_writer.writerow([index, "document", f"{name}.jpg"])


def converting_ubiai(data):
    train, test = train_test_split(data, train_size=0.8)
    train_writer = csv.writer(open("data/train.csv", "w", newline=""))
    train_set = "train"
    index = 0
    for line in train:
        index += 1
        processing_image(train_set, line, train_writer, index)
    test_writer = csv.writer(open("data/test.csv", "w", newline=""))
    train_set = "test"
    index = 0
    for line in test:
        index += 1
        processing_image(train_set, line, test_writer, index)


def slipt_wildreceipt():
    for i in ['train', 'test']:
        with open(f'wildreceipt/wildreceipt_{i}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                file_name = line.split('", "')[0].split('": "')[-1]
                try:
                    shutil.move(f'wildreceipt/{file_name}', f'wildreceipt/{i}/')
                except Exception as e:
                    print(e)


def get_class_list():
    file = '../mmocr2/tests/wildreceipt/annotate.json'
    data = json.load(open(file, 'r'))
    labels = []
    index = 0
    seen = set()
    with open('../mmocr2/tests/wildreceipt/class_list.txt', 'w') as f:
        for line in data:
            if line['annotation']:
                bboxs = line['annotation']
                if bboxs:
                    for item in bboxs:
                        label = item['label']
                        if label not in seen:
                            seen.add(label)
                            labels.append(f'{index}  {label}\n')
                            index += 1
        f.writelines([label for label in labels])


if __name__ == '__main__':
    file = 'data/annotate.json'
    data = json.load(open(file, 'r'))
    converting_ubiai(data)
    # slipt_wildreceipt()
    # get_class_list()