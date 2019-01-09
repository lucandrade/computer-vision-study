# USAGE
# python compare.py

# import the necessary packages
import matplotlib
matplotlib.use('TKAgg')

from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(fig, imageA, imageB, title, row):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# show first image
	ax = fig.add_subplot(3, 2, row)
	ax.set_title("MSE: %.2f, SSIM: %.2f" % (m, s))
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(3, 2, row + 1)
	ax.set_title(title)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("images/jp_gates_original.png")
contrast = cv2.imread("images/jp_gates_contrast.png")
shopped = cv2.imread("images/jp_gates_photoshopped.png")

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

def show_original_images():
	# initialize the figure
	fig = plt.figure("Images")
	images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)

	# loop over the images
	for (i, (name, image)) in enumerate(images):
		# show the image
		ax = fig.add_subplot(1, 3, i + 1)
		ax.set_title(name)
		plt.imshow(image, cmap = plt.cm.gray)
		plt.axis("off")

	# show the figure
	plt.show()

show_original_images()

fig = plt.figure("Comparison")

# compare the images
compare_images(fig, original, original, "Original", 1)
compare_images(fig, original, contrast, "Contrast", 3)
compare_images(fig, original, shopped, "Photoshopped", 5)

plt.show()