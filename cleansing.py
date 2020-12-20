import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 水増しのための関数を定義
def scratch_image(img, flip=True, thr=True, filt=True, resize=True, erode=True):
	#methods = [flip, thr, filt, resize, erode]
	img_size = img.shape
	filter1 = np.ones((3,3))

	images = []

	if flip == True:
		images.append(cv2.flip(img, 1))
	if thr == True:
		images.append(cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)[1])
	if filt == True:
		images.append(cv2.GaussianBlur(img, (5,5), 0))
	if resize == True:
		images.append(cv2.resize(cv2.resize(img, (img_size[1] // 5, img_size[0] //5)), (img_size[1], img_size[0])))
	if erode == True:
		images.append(cv2.erode(img, filter1))

	return images

# 読み込み画像パス
datadir = "./images/"
categories = ["Yorkshire_terrier"]
#categories = ["French_bulldog", "Chihuahua", "Golden_retriever", "Maltese_dog", "Miniature_Dachshund", "Saint_Bernard", "Shiba", "Shih_Tzu", "Toypoodle", "Yorkshire_terrier"]

dog_imgs = []
for category in categories:
	path = os.path.join(datadir, category)
	#隠しファイルは読まないため
	data_list = [data for data in os.listdir(path) if not data.startswith('.')]
	for data in data_list:
		data = path + "/" + data
		dog_imgs.append(data)

# 画像の水増し
scratch_images = []
for dog_img in dog_imgs:
	img = cv2.imread(dog_img)
	scratch_images += scratch_image(img)

# 画像を保存するフォルダーを作成
if not os.path.exists("Yorkshire_terrier"):
	os.mkdir("Yorkshire_terrier")
for num, im in enumerate(scratch_images):
	# まず保存先のディレクトリ"scratch_images/"を指定、番号を付けて保存
	cv2.imwrite("Yorkshire_terrier/" + str(1000 + num) + ".jpg", im)
