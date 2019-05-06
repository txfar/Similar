#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageFont
import numpy as np
import cv2
import os
from pylab import *



def make_regalur_image(img, size = (256, 256)):
	return img.resize(size).convert('RGB')

def split_image(img, part_size = (64, 64)):
	w, h = img.size
	pw, ph = part_size
	
	assert w % pw == h % ph == 0
	
	return [img.crop((i, j, i+pw, j+ph)).copy() \
				for i in range(0, w, pw) \
				for j in range(0, h, ph)]

def hist_similar(lh, rh):
	assert len(lh) == len(rh)
	return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def calc_similar(li, ri):
#	return hist_similar(li.histogram(), ri.histogram())
	return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0
			

def calc_similar_by_path(lf, rf):
	li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
	return calc_similar(li, ri)

def make_doc_data(lf, rf):
	li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
	li.save(lf + '_regalur.png')
	ri.save(rf + '_regalur.png')
	fd = open('stat.csv', 'w')
	fd.write('\n'.join(l + ',' + r for l, r in zip(map(str, li.histogram()), map(str, ri.histogram()))))

def file_name(file_dir):
	L = []
	for root, dirs, files in os.walk(file_dir):
		for file in files:
			L.append(os.path.join(root, file))#是否有必要筛选图片文件类型
	return L
#	print >>fd, '\n'
#	fd.write(','.join(map(str, ri.histogram())))
	fd.close()
	import ImageDraw
	li = li.convert('RGB')
	draw = ImageDraw.Draw(li)
	for i in range(0, 256, 64):
		draw.line((0, i, 256, i), fill = '#ff0000')
		draw.line((i, 0, i, 256), fill = '#ff0000')
	li.save(lf + '_lines.png')
	
#    利用calc_similar_by_path(lr,rl)测算每个request与gallery的相似度；返回一个相似度的m*n维数组记录相似度，利用排序数组来解决位置对应问题（后续为了提高运算速度，可以适当提高维数进行文件名的记录）

if __name__ == '__main__':
	path = r'test/TEST%d/%d.JPG'


	file_query_dir='request/query'
	file_gallery_dir='request/gallery'
	Query_name=[]
	Gallery_name=[]
	similarity=[]
	sim=[]
	follow_seq=[]
	seq=[]
	pic=[]
	seq_str=''
	Query_name=file_name(file_query_dir)
	Gallery_name=file_name(file_gallery_dir)
	query_num=len(Query_name)
	gallery_num=len(Gallery_name)

	for i in range(0, query_num):
		for j in range(0,gallery_num):
			sim.append(calc_similar_by_path(Query_name[i],Gallery_name[j]))
			seq.append(Gallery_name[j])
		similarity.append(sim)
		follow_seq.append(seq)
		sim=[]
		seq=[]

	for i in range (0,query_num):
		for j in range (0,gallery_num):
			maxj=j
			for k in range (j,gallery_num):
				if similarity[i][k]>similarity[i][maxj]:
					maxj=k
			if j != maxj :
				similarity[i][maxj]=similarity[i][j]+similarity[i][maxj]
				seq_str=follow_seq[i][j]
				similarity[i][j]=similarity[i][maxj]-similarity[i][j]
				follow_seq[i][j]=follow_seq[i][maxj]
				similarity[i][maxj]=similarity[i][maxj]-similarity[i][j]
				follow_seq[i][maxj]=seq_str

	for i in range(0,query_num):
		print('%s:         '%Query_name[i])
		for j in range (0,gallery_num):
			print('%s  %.3f%%'%(follow_seq[i][j],similarity[i][j]*100))



	fig = plt.figure()
	for i in range(0,query_num):
		print('%s:         '%Query_name[i])
		img = cv2.imread(Query_name[i],cv2.IMREAD_COLOR);
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		plt.subplot(5,11,i*11+1)
		plt.imshow(img)
		(filepath,tempfilename) = os.path.split(Query_name[i])
		(filename,extension) = os.path.splitext(tempfilename)
		title(filename,fontsize=8)
		for j in range (0,gallery_num):
			img = cv2.imread(follow_seq[i][j],cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			plt.subplot(5,11,i*11+j+2)
			plt.imshow(img)
			(filepath,tempfilename) = os.path.split(follow_seq[i][j])
			(filename,extension) = os.path.splitext(tempfilename)
			title(filename,fontsize=8)
			plt.xlabel(round(similarity[i][j]*100,3),fontsize=8)
	plt.show()


			#cv2.imshow()
			#cv2.imwrite()

#	make_doc_data('test/TEST4/1.JPG', 'test/TEST4/2.JPG')

