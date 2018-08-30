from random import random

class Cell:
	def __init__(self, inputSize, learningRate):
		self.weights = [0] * inputSize
		self.output = 0
		self.learningRate = learningRate

	def initCell(self):
		for i in range(len(self.weights)):
			self.weights[i] = random()
		self.output = 0

	def calculateOutput(self, image):
		inputVector = image
		self.output = 0
		for i in range(len(inputVector)):
			self.output += inputVector[i] * self.weights[i]
		self.output /= len(inputVector)

	def updateWeights(self, error, inputVector):
		for i in range(len(self.weights)):
			self.weights[i] += inputVector[i] * error * self.learningRate



class SLP:
	def __init__(self, layerSize, inputSize, learningRate):
		self.layer = []
		for i in range(layerSize):
			self.layer.append(Cell(inputSize, learningRate))
		for i in range(layerSize):
			self.layer[i].initCell()

	def train(self, images):
		imageCount = len(images)
		correct = 0
		for imageIndex in range(len(images)):
			image = images[imageIndex]
			expectedOut = [0] * len(self.layer)
			expectedOut[image.label] = 1
			for i in range(len(self.layer)):
				cell = self.layer[i]
				cell.calculateOutput(image.imageVector)
				diff = expectedOut[i] - cell.output
				cell.updateWeights(diff, image.imageVector)
			prediction = 0
			for i in range(len(self.layer)):
				if self.layer[i].output > self.layer[prediction].output:
					prediction = i
			if prediction == image.label:
				correct += 1
			print("Training: {:d}/{:d}    Success: {:.2f}%\r".format(imageIndex + 1, imageCount, 100*(correct/(imageIndex + 1))), end="")
		print("")

	def test(self, images):
		imageCount = len(images)
		correct = 0
		for imageIndex in range(len(images)):
			image = images[imageIndex]
			expectedOut = [0] * len(self.layer)
			expectedOut[image.label] = 1
			for cell in self.layer:
				cell.calculateOutput(image.imageVector)
			prediction = 0
			for i in range(len(self.layer)):
				if self.layer[i].output > self.layer[prediction].output:
					prediction = i
			if prediction == image.label:
				correct += 1
			print("Testing: {:d}/{:d}    Success: {:.2f}%\r".format(imageIndex + 1, imageCount, 100*(correct/(imageIndex + 1))), end="")
		print("")