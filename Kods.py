#!/usr/bin/env python3

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read(path):
	img = cv2.imread(path)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def plot(img, title, save=os.getenv('SCRIPT_SAVE_IMG') is not None):
	if save:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(f'Figūras/{ title }.jpg', img)
		return

	plt.imshow(img)

	plt.title(title)
	plt.axis(False)

	plt.show()


def gaussian_thresh(img, kernel=15, c=0):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
	thresh = cv2.THRESH_BINARY

	output = cv2.adaptiveThreshold(img, 255, method, thresh, kernel, c)
	return cv2.applyColorMap(output, cv2.COLORMAP_SUMMER)

def k_means(img, k=5, iter=20, eps=1):
	data = img.reshape(-1, 3).astype(np.float32)

	criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, iter, eps)
	origin = cv2.KMEANS_PP_CENTERS

	_, labels, centers = cv2.kmeans(data, k, None, criteria, iter, origin)
	labels = labels.reshape(img.shape[:2])

	cmap   = plt.get_cmap('summer')
	output = np.empty_like(img)

	for i in range(k):
		rgb = cmap(i / (k - 1))[:3]
		output[labels == i] = [np.uint8(c * 255) for c in rgb]

	return output


A = read('Bildes/Putni.jpg')
plot(A, 'Putni', save=False)

B = gaussian_thresh(A)
plot(B, 'Putni, Gausa adaptīvā sliekšņošana')

B = k_means(A, k=2)
plot(B, 'Putni, K-Means k=2')

B = k_means(A, k=5)
plot(B, 'Putni, K-Means k=5')


A = read('Bildes/Kaķis.jpg')
plot(A, 'Kaķis', save=False)

B = gaussian_thresh(A)
plot(B, 'Kaķis, Gausa adaptīvā sliekšņošana')

B = k_means(A, k=2)
plot(B, 'Kaķis, K-Means k=2')

B = k_means(A, k=5)
plot(B, 'Kaķis, K-Means k=5')


A = read('Bildes/Dzīvnieki.jpg')
plot(A, 'Dzīvnieki', save=False)

B = gaussian_thresh(A)
plot(B, 'Dzīvnieki, Gausa adaptīvā sliekšņošana')

B = k_means(A, k=2)
plot(B, 'Dzīvnieki, K-Means k=2')

B = k_means(A, k=5)
plot(B, 'Dzīvnieki, K-Means k=5')

