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


def sort_labels(labels, centers):
	intens = np.mean(centers, axis=1)
	sorted = np.argsort(intens)

	table = np.empty(len(intens), dtype=labels.dtype)
	table[sorted] = np.arange(len(intens))

	return table[labels]

def draw_clusters(labels, clusters, cmap=plt.get_cmap('jet')):
	output = np.empty((*labels.shape, 3), dtype=np.uint8)
	for i in range(clusters):
		rgb = cmap(i / clusters)[:3]
		output[labels == i] = [np.uint8(c * 255) for c in rgb]

	return output


def gaussian_mixture(img, clusters=5, iter=20, eps=1):
	data = img.reshape(-1, 3).astype(np.float32)

	criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, iter, eps)

	em = cv2.ml.EM.create()
	em.setClustersNumber(clusters)
	em.setTermCriteria(criteria)

	_, _, labels, _ = em.trainEM(data)
	centers = em.getMeans()

	labels = labels.reshape(img.shape[:2])
	labels = sort_labels(labels, centers)

	return draw_clusters(labels, clusters)

def k_means(img, clusters=5, iter=20, eps=1):
	data = img.reshape(-1, 3).astype(np.float32)

	criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, iter, eps)
	origin = cv2.KMEANS_PP_CENTERS

	_, labels, centers = cv2.kmeans(data, clusters, None, criteria, iter, origin)

	labels = labels.reshape(img.shape[:2])
	labels = sort_labels(labels, centers)

	return draw_clusters(labels, clusters)


A = read('Bildes/Putni.jpg')
plot(A, 'Putni', save=False)

B = gaussian_mixture(A)
plot(B, 'Putni - Gausa adaptīvā sliekšņošana')
B = k_means(A)
plot(B, 'Putni - K-Means')

A = read('Bildes/Kaķis apģērbā.jpg')
plot(A, 'Kaķis apģērbā', save=False)

B = gaussian_mixture(A)
plot(B, 'Kaķis apģērbā - Gausa adaptīvā sliekšņošana')
B = k_means(A)
plot(B, 'Kaķis apģērbā - K-Means')

A = read('Bildes/Dzīvnieki.jpg')
plot(A, 'Dzīvnieki', save=False)

B = gaussian_mixture(A)
plot(B, 'Dzīvnieki - Gausa adaptīvā sliekšņošana')
B = k_means(A)
plot(B, 'Dzīvnieki - K-Means')

