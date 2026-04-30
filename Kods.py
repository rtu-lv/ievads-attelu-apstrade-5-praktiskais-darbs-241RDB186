#!/usr/bin/env python3

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read(path):
	img = cv2.imread(path)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def plot(img, title, save=os.getenv("SCRIPT_SAVE_IMG") is not None):
	if save:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(f"Figūras/{ title }.jpg", img)
		return

	plt.imshow(img, cmap="gray")
	plt.title(title)
	plt.axis(False)

	plt.show()

