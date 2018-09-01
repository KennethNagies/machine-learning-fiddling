import os

class MNIST_Image:
	TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "./TestingImages")
	TEST_LABEL_PATH = os.path.join(os.path.dirname(__file__), "./TestingLabels")
	TRAIN_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "./TrainingImages")
	TRAIN_LABEL_PATH = os.path.join(os.path.dirname(__file__), "./TrainingLabels")

	def __init__(self, width, height, imageVector, label):
		self.width = width
		self.height = height
		self.imageVector = [0] * (width * height)
		for i in range(width * height):
			self.imageVector[i] = imageVector[i]
		self.label = label
		self.maxValue = 255

	# converts all pixels in the image to be 0 or 1 instead
	# of standard 0-255 grayscale. Helps with some classification
	# methods.
	def simplify(self):
		for i in range(self.height * self.width):
			if self.imageVector[i] > 0:
				self.imageVector[i] = 1
		self.maxValue = 1

	# Returns an array of training MNIST_Image objects
	def getTrainingData():
		print("Reading Training Images:")
		return MNIST_Image.readDataFile(MNIST_Image.TRAIN_IMAGE_PATH, MNIST_Image.TRAIN_LABEL_PATH)

	# Returns an array of testing MNIST_Image objects
	def getTestingData():
		print("Reading Testing Images:")
		return MNIST_Image.readDataFile(MNIST_Image.TEST_IMAGE_PATH, MNIST_Image.TEST_LABEL_PATH)

	# Reads the Image and label files at the paths given and returns an array of the resulting
	# MNIST_Image Objects
	def readDataFile(imageFileName, labelFileName):
		images = []
		imageFile = open(imageFileName, "rb")
		labelFile = open(labelFileName, "rb")
		try:
			# This is the magic number that doesnt matter for our purposes
			imageFile.read(4)
			labelFile.read(8)
			imageCount = int.from_bytes(imageFile.read(4), byteorder='big')
			rows = int.from_bytes(imageFile.read(4), byteorder='big')
			cols = int.from_bytes(imageFile.read(4), byteorder='big')
			imageBuffer = imageFile.read(rows * cols * imageCount)
			labelBuffer = labelFile.read(imageCount)
			labelIndex = 0
			byteIndex = 0
			imageIndex = 0
			for byte in imageBuffer:
				if byteIndex == 0:
					tempImage = [0] * (rows * cols)
				tempImage[byteIndex] = byte
				byteIndex = (byteIndex + 1) % (rows * cols)
				if byteIndex == 0:
					label = labelBuffer[labelIndex]
					labelIndex += 1
					images.append(MNIST_Image(cols, rows, tempImage, label))
					print(str((labelIndex)) + "/" + str(imageCount) + "\r", end="")
			print(str(imageCount) + "/" + str(imageCount))
		finally:
			imageFile.close()
			labelFile.close()
		return images

	# Prints an image to the Console with '#'s for non zeroes and ' 's for zeroes along with the label.
	def printImage(image):
		print(image.label)
		print('-' * (image.width + 2))
		for row in range(image.height):
			rowArr = [""] * (image.width + 2)
			rowArr[0] = "|"
			rowArr[-1] = "|"
			for col in range(image.width):
				if image.imageVector[row * image.width + col] == 0:
					rowArr[col + 1] = " "
				else:
					rowArr[col + 1] = "#"
			print(''.join(rowArr))
		print('-' * (image.width + 2))