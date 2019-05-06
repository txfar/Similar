# -*-coding:utf-8 -*-

from PIL import Image
import numpy as np
import cv2
import os
from pylab import *


def loadImage(image):
    # 读取图片
    im = cv2.imread(image, cv2.IMREAD_COLOR);
    # im = Image.open(image)
    # im = im.convert("RGB")
    data = np.array(im)
    return data


def subtraction(p, q):
    s = p - q
    return s


def similar_by_path(p, q):
    data1 = loadImage(p)
    data2 = loadImage(q)
    s = data1 + data2
    s = sum(s)
    return s


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))  # 是否有必要筛选图片文件类型
    return L


# li = li.convert('RGB')
# draw = ImageDraw.Draw(li)
# for i in range(0, 256, 64):
# 	draw.line((0, i, 256, i), fill = '#ff0000')
# 	draw.line((i, 0, i, 256), fill = '#ff0000')
# li.save(lf + '_lines.png')

if __name__ == "__main__":
    file_query_dir = 'request/query'
    file_gallery_dir = 'request/gallery'
    Query_name = []
    Gallery_name = []
    similarity = []
    sim = []
    follow_seq = []
    seq = []
    pic = []
    seq_str = ''
    Query_name = file_name(file_query_dir)
    Gallery_name = file_name(file_gallery_dir)
    query_num = len(Query_name)
    gallery_num = len(Gallery_name)

    for i in range(0, query_num):
        for j in range(0, gallery_num):
            sim.append(similar_by_path(Query_name[i], Gallery_name[j]))
            seq.append(Gallery_name[j])
        similarity.append(sim)
        follow_seq.append(seq)
        sim = []
        seq = []

    for i in range(0, query_num):
        for j in range(0, gallery_num):
            minj = j
            for k in range(j, gallery_num):
                if similarity[i][k] < similarity[i][minj]:
                    minj = k
            similarity[i][minj] = similarity[i][j] + similarity[i][minj]
            seq_str = follow_seq[i][j]
            similarity[i][j] = similarity[i][minj] - similarity[i][j]
            follow_seq[i][j] = follow_seq[i][minj]
            similarity[i][minj] = similarity[i][minj] - similarity[i][j]
            follow_seq[i][minj] = seq_str

    for i in range(0, query_num):
        print('%s:         ' % Query_name[i])
        for j in range(0, gallery_num):
            print('%s  %.0f' % (follow_seq[i][j], similarity[i][j]))

    fig = plt.figure()
    for i in range(0, query_num):
        print('%s:         ' % Query_name[i])
        img = cv2.imread(Query_name[i], cv2.IMREAD_COLOR);
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(5, 11, i * 11 + 1)
        plt.imshow(img)
        (filepath, tempfilename) = os.path.split(Query_name[i])
        (filename, extension) = os.path.splitext(tempfilename)
        title(filename, fontsize=8)
        for j in range(0, gallery_num):
            img = cv2.imread(follow_seq[i][j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(5, 11, i * 11 + j + 2)
            plt.imshow(img)
            (filepath, tempfilename) = os.path.split(follow_seq[i][j])
            (filename, extension) = os.path.splitext(tempfilename)
            title(filename, fontsize=8)
            plt.xlabel(round(similarity[i][j] * 100, 3), fontsize=8)
        plt.show()

# image1 = "D:/02workspace/python/Pedestrian/similar_degree/request/gallery/0650_c3_034562.jpg"
# data1 = loadImage(image1)
#
# image2 = "D:/02workspace/python/Pedestrian/similar_degree/request/query/0650_c2_031237.jpg"
# data2 = loadImage(image2)
#
# data = np.sum(subtraction(data1, data2))
# print(data)
