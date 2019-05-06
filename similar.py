from numpy import average, dot, linalg
from PIL import Image
import numpy as np
import cv2
import os
from pylab import *


# class SQList:
#     def __init__(self, lis=None):
#         self.r = lis
#
#     def swap(self, i, j):
#         """定义一个交换元素的方法，方便后面调用。"""
#         temp = self.r[i]
#         self.r[i] = self.r[j]
#         self.r[j] = temp
#
#     def quick_sort(self):
#         """调用入口"""
#         self.qsort(0, len(self.r)-1)
#
#     def qsort(self, low, high):
#         """递归调用"""
#         if low < high:
#             pivot = self.partition(low, high)
#             self.qsort(low, pivot-1)
#             self.qsort(pivot+1, high)
#
#     def partition(self, low, high):
#         """
#         快速排序的核心代码。
#         其实就是将选取的pivot_key不断交换，将比它小的换到左边，将比它大的换到右边。
#         它自己也在交换中不断变换自己的位置，直到完成所有的交换为止。
#         但在函数调用的过程中，pivot_key的值始终不变。
#         :param low:左边界下标
#         :param high:右边界下标
#         :return:分完左右区后pivot_key所在位置的下标
#         """
#         lis = self.r
#         pivot_key = lis[low]
#         while low < high:
#             while low < high and lis[high] >= pivot_key:
#                 high -= 1
#             self.swap(low, high)
#             while low < high and lis[low] <= pivot_key:
#                 low += 1
#             self.swap(low, high)
#         return low
#
#     def __str__(self):
#         ret = ""
#         for i in self.r:
#             ret += " %s" % i
#         return ret

# 对图片进行统一化处理

def get_thum(image, size=(64, 64)):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    return image


# 计算图片的余弦距离

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)

    image2 = get_thum(image2)

    images = [image1, image2]

    vectors = []

    norms = []

    for image in images:

        vector = []

        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))

        vectors.append(vector)

        # linalg=linear（线性）+algebra（代数），norm则表示范数

        # 求图片的范数？？

        norms.append(linalg.norm(vector, 2))

    a, b = vectors

    a_norm, b_norm = norms

    # dot返回的是点积，对二维数组（矩阵）进行计算

    res = dot(a / a_norm, b / b_norm)

    return res


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))  # 是否有必要筛选图片文件类型
    return L


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
            s = image_similarity_vectors_via_numpy(Image.open(Query_name[i]), Image.open(Gallery_name[j]))
            sim.append(s)
            seq.append(Gallery_name[j])
        similarity.append(sim)
        follow_seq.append(seq)
        sim = []
        seq = []

    # for i in range(0, query_num):
    #     for j in range(0, gallery_num):
    #         sqlist = SQList(similarity[i][j])
    #         sqlist.quick_sort()
    #         print(sqlist)

    for i in range(0, query_num):
        for j in range(0, gallery_num):
            maxj = j
            for k in range(j, gallery_num):
                if similarity[i][k] > similarity[i][maxj]:
                    maxj = k
            if maxj != j:
                similarity[i][maxj] = similarity[i][j] + similarity[i][maxj]
                seq_str = follow_seq[i][j]
                similarity[i][j] = similarity[i][maxj] - similarity[i][j]
                follow_seq[i][j] = follow_seq[i][maxj]
                similarity[i][maxj] = similarity[i][maxj] - similarity[i][j]
                follow_seq[i][maxj] = seq_str

for i in range(0, query_num):
    print('%s:         ' % Query_name[i])
    for j in range(0, gallery_num):
        print('%s  %.3f' % (follow_seq[i][j], similarity[i][j]))

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

# image1 = Image.open('D:/02workspace/python/Pedestrian/similar_degree/request/gallery/0650_c3_034562.jpg')
#
# image2 = Image.open('D:/02workspace/python/Pedestrian/similar_degree/request/query/0650_c2_031237.jpg')

# cosin = image_similarity_vectors_via_numpy(image1, image2)

# print('图片余弦相似度', cosin)
