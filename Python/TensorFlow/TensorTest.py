import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../MNIST"))
from MNIST import MNIST_Image
import pandas

def dataFrameFromImages(images):
	imageMatrix = [[0] * len(images) for i in range(len(images[0].imageVector))]
	for pixelIndex in range(len(imageMatrix)):
		for imageIndex in range(len(imageMatrix[0])):
			imageMatrix[pixelIndex][imageIndex] = images[imageIndex].imageVector[pixelIndex]
	dfDict = {}
	for pixelIndex in range(len(imageMatrix)):
		dfDict[str(pixelIndex)] = imageMatrix[pixelIndex]
	labels = [0] * len(images)
	for imageIndex in range(len(images)):
		labels[imageIndex] = images[imageIndex].label
	dfDict["label"] = labels
	return pandas.DataFrame(dfDict)

def main():
	testingImages = MNIST_Image.getTestingData()
	testingDataFrame = dataFrameFromImages(testingImages)
	print(testingDataFrame.head())
	print(testingImages[1].imageVector[93 : 102])

if __name__=="__main__":
	main()